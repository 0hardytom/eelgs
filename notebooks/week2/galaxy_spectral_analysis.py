# -*- coding: utf-8 -*-
"""
This script provides a workflow for manually extracting and analyzing a galaxy 
spectrum from a VLT/MUSE datacube using MPDAF.

To use this script:
1.  Modify the parameters in the 'USER INPUT' section below to set the path to your
    datacube, the target galaxy's coordinates (RA, Dec), the aperture radius for 
    extraction, and the estimated redshift.
2.  Run the script from your terminal:
    python galaxy_spectral_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from astropy.units import u
import sys

# ==============================================================================
# USER INPUT: Define Target and Extraction Parameters
# ==============================================================================

# Path to the MUSE datacube from the project root directory
# IMPORTANT: This script assumes you are running it from the project's root directory.
CUBE_PATH = 'cubes/macs0159m34_COMBINED_CUBE_MED_FINAL.fits'

# Target coordinates in degrees (J2000)
RA_DEG = 24.990458 
DEC_DEG = -34.131583

# Aperture radius in arcseconds for the spectral extraction
RADIUS_ARCSEC = 1.2

# Estimated redshift of the galaxy
Z = 0.405

# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_galaxy_spectrum(cube_path, ra, dec, radius, z):
    """
    Loads a MUSE cube, extracts a spectrum at a given position, plots it,
    fits key emission lines, and calculates diagnostic line ratios.
    """
    # 1. Load the Datacube
    print(f"Loading datacube: {cube_path}...")
    try:
        cube = Cube(cube_path)
        print(f"Successfully loaded cube. Dimensions: {cube.shape}")
    except FileNotFoundError:
        print(f"Error: Datacube not found at '{cube_path}'")
        print("Please ensure the path is correct and you are running this script from the project's root directory.")
        sys.exit(1)

    # 2. Extract the 1D Spectrum
    center = (dec, ra)  # (Dec, RA) order for MPDAF
    print(f"\nExtracting spectrum at (RA, Dec) = ({ra:.6f}, {dec:.6f}) with a {radius}\" radius aperture.")
    spec = cube.aperture(center, radius, is_sum=True)
    print("Extraction complete.")

    # 3. Plot the Extracted Spectrum
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    spec.plot(ax=ax1, title=f"Extracted Spectrum at (RA={ra:.4f}, Dec={dec:.4f})")
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylabel(f"Flux ({spec.unit})")
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig1.canvas.manager.set_window_title('Extracted Spectrum')


    # 4. Measure Emission Line Properties
    print("\nFitting emission lines...")
    # De-redshift the spectrum to the rest frame by manually adjusting the wavelength solution
    spec_rest = spec.copy()
    spec_rest.wave.set_crval(spec_rest.wave.get_crval() / (1 + z))
    spec_rest.wave.set_step(spec_rest.wave.get_step() / (1 + z))

    # Define emission lines (rest-frame wavelengths in Angstroms)
    lines = {
        'Hbeta': 4861.33,
        'OIII_5007': 5006.84,
        'Halpha': 6562.80,
        'NII_6583': 6583.45
    }

    line_fits = {}
    # Fit each line with a Gaussian
    for name, wavelength in lines.items():
        try:
            # Fit the line within a +/- 15 Angstrom window
            fit = spec_rest.gauss_fit(lmin=(wavelength - 15), lmax=(wavelength + 15), plot=False)
            line_fits[name] = fit
            print(f"- {name}: Flux = {fit.flux.value:.2e} +/- {fit.flux.error:.2e} {fit.flux.unit}")
        except Exception as e:
            print(f"- Could not fit {name}. Error: {e}")
            line_fits[name] = None

    # Plot the fits for visual inspection
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Gaussian Fits to Emission Lines', fontsize=16)
    fig2.canvas.manager.set_window_title('Emission Line Fits')

    for ax, name in zip(axes.ravel(), lines.keys()):
        fit = line_fits.get(name)
        if fit is not None:
            fit.plot(ax=ax, title=name, fit_kws={'color': 'red', 'lw': 2}, data_kws={'color': 'black', 'alpha': 0.7})
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.set_title(f"{name} (Fit Failed)")
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 5. Calculate Line Ratios
    print("\nCalculating line ratios...")

    # Calculate OIII/Hbeta ratio
    oiii_fit = line_fits.get('OIII_5007')
    hbeta_fit = line_fits.get('Hbeta')
    if oiii_fit and hbeta_fit and hbeta_fit.flux.value > 0:
        oiii_hbeta_ratio = oiii_fit.flux / hbeta_fit.flux
        print(f"[OIII]/Hβ = {oiii_hbeta_ratio.value:.3f} +/- {oiii_hbeta_ratio.error:.3f}")
        print(f"log([OIII]/Hβ) = {np.log10(oiii_hbeta_ratio.value):.3f}\n")
    else:
        print("[OIII]/Hβ could not be calculated.\n")

    # Calculate NII/Halpha ratio
    nii_fit = line_fits.get('NII_6583')
    halpha_fit = line_fits.get('Halpha')
    if nii_fit and halpha_fit and halpha_fit.flux.value > 0:
        nii_halpha_ratio = nii_fit.flux / halpha_fit.flux
        print(f"[NII]/Hα = {nii_halpha_ratio.value:.3f} +/- {nii_halpha_ratio.error:.3f}")
        print(f"log([NII]/Hα) = {np.log10(nii_halpha_ratio.value):.3f}\n")
    else:
        print("[NII]/Hα could not be calculated.\n")
        
    # Display plots
    print("Displaying plots. Close the plot windows to exit the script.")
    plt.show()


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # Configure plot style
    try:
        plt.style.use('seaborn-v0_8-colorblind')
    except IOError:
        print("Seaborn style not found, using default.")

    # Run the analysis
    analyze_galaxy_spectrum(
        cube_path=CUBE_PATH,
        ra=RA_DEG,
        dec=DEC_DEG,
        radius=RADIUS_ARCSEC,
        z=Z
    )
