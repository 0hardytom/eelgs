
import pandas as pd
import numpy as np
from mpdaf.obj import Cube, Spectrum
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import os
from tqdm import tqdm

def process_spectrum(row, cube, common_wave_grid):
    """
    Processes a single spectrum from a FITS cube based on a row of metadata.
    Resamples the spectrum onto a common, predefined wavelength grid.
    """
    # try:
    # Create a circular aperture and extract the spectrum
    print(f'processing {row['cluster_key']},{row['object_id']}')
    # spec = cube.aperture((row['Y_PEAK_SN'], row['X_PEAK_SN']), 2, unit_center=None)
    spec = cube.aperture((row['dec'], row['ra']), 2)

    # Go to rest frame
    z = row['z']
    observed_wave = spec.wave.coord()
    rest_wave = observed_wave / (1 + z)
    
    flux = spec.data
    
    # Manually calculate the continuum using a rolling median
    flux_series = pd.Series(flux)
    continuum_flux = flux_series.rolling(window=26, center=True, min_periods=1).median().values

    # Avoid division by zero or near-zero
    continuum_flux[continuum_flux < 1e-6] = 1e-6
    normalized_flux = spec.data / continuum_flux

    # Manually resample to the common grid using interpolation
    # Use bounds_error=False to allow extrapolation, fill_value=np.nan for areas outside the original spectrum
    interp_func = interp1d(rest_wave, normalized_flux, kind='linear', bounds_error=False, fill_value=np.nan)
    resampled_flux = interp_func(common_wave_grid)
    
    return resampled_flux

    # except Exception as e:
    #     print(f"Could not process ID {row['ID']}: {e}")
    #     return None

def main():
    """
    Main function to run the co-addition preparation script.
    """
    # Define paths
    csv_path = 'PEAS.csv'
    base_fits_path = '/Volumes/Expansion/exp_thardy/'
    output_fits_path = 'coadd_peas_spectra.fits'

    # Read the CSV
    print(f"Reading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter out bad redshift values
    df = df[df['z'] > 0]

    # Define the common rest-frame wavelength grid for all spectra
    # This ensures all output spectra have the same length and wavelength points.
    wave_min_common = 620.0
    wave_max_common = 9300.0
    common_wave_grid = np.arange(wave_min_common, wave_max_common + 2.0, 2.0)

    all_spectra = []
    
    print("Processing spectra...")
    
    # Group by cube file to avoid reopening files
    grouped = df.groupby(['source_dir', 'cluster_key'])
    
    for (directory, key), group in tqdm(grouped, total=len(grouped)):
        fits_path = os.path.join(base_fits_path, directory, f"{key}_COMBINED_CUBE_MED_FINAL.fits")
        
        if not os.path.exists(fits_path):
            print(f"File not found: {fits_path}")
            continue
            
        # try:
        cube = Cube(fits_path)
        for _, row in group.iterrows():
            resampled_flux = process_spectrum(row, cube, common_wave_grid)
            
            if resampled_flux is not None:
                # Store as a dictionary
                all_spectra.append({
                    'ID': row['object_id'],
                    'Redshift': row['z'],
                    'Wavelength': common_wave_grid,
                    'Flux': resampled_flux
                })
        # except Exception as e:
            # print(f"Could not process cube {fits_path}: {e}")

    if not all_spectra:
        print("No spectra were successfully processed. Exiting.")
        return

    # Create an Astropy Table
    # All arrays now have the same length
    table = Table(all_spectra)

    # Save to a FITS file
    print(f"Saving master table to {output_fits_path}")
    hdu = fits.BinTableHDU(table)
    hdu.writeto(output_fits_path, overwrite=True)
    print("Done.")

if __name__ == '__main__':
    main()
