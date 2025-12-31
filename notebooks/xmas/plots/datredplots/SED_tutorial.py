# SED_tutorial.py
#
# A step-by-step guide to deriving galaxy stellar mass from an MPDAF spectrum
# using the Prospector fitting code.
#
# This script is intended as a tutorial. You will need to:
#   1. Install the required libraries:
#      pip install mpdaf prospector python-fsps sedpy dynesty
#   2. Change the 'spec_file' variable to point to your own data.
#   3. Change the 'z' variable to the correct redshift for your galaxy.
#   4. Be aware that the fitting process (Step 5) is computationally
#      intensive and can take several hours to run.

import numpy as np
import os
import matplotlib.pyplot as plt

# --- Configuration ---
# IMPORTANT: Update these values for your specific target
SPEC_FILE = 'spec/macs0027p26b/6d55m56pt13s26d16m02pt84s_rest_spec.fits' # Example path
REDSHIFT = 0.7  # Example redshift
RESULTS_FILE = "galaxy_fit_results.h5"

def run_prospector_fit():
    """
    Main function to run the full SED fitting workflow.
    """
    
    # =========================================================================
    # Step 2: Data Preparation with MPDAF
    # =========================================================================
    print("--- Step 2: Preparing data with MPDAF ---")
    
    from mpdaf.obj import Spectrum
    from astropy.cosmology import Planck15 as cosmo
    from astropy import units as u

    if not os.path.exists(SPEC_FILE):
        print(f"Error: Spectrum file not found at '{SPEC_FILE}'")
        print("Please update the SPEC_FILE variable in this script.")
        return

    # --- 1. Load the spectrum ---
    spec = Spectrum(SPEC_FILE)

    # --- 2. Extract Wavelength, Flux, and Uncertainty ---
    wave_rest = spec.wave.coord()  # Angstroms
    flux = spec.data               # erg/s/cm^2/Angstrom (check units)
    flux_unc = np.sqrt(spec.var)   # Get std deviation from variance

    # --- 3. Convert Flux Units to Maggies ---
    # Prospector's models operate in "maggies", a linear flux unit.
    d_lum = cosmo.luminosity_distance(REDSHIFT).to(u.cm).value 
    c = u.lightspeed.to(u.AA / u.s).value

    flux_nu = flux * (wave_rest**2 / c)
    unc_nu = flux_unc * (wave_rest**2 / c)
    flux_jy = flux_nu / 1e-23
    unc_jy = unc_nu / 1e-23

    # Scale by luminosity distance to get intrinsic maggies for the model
    maggies = (flux_jy / 3631.0) * (4 * np.pi * d_lum**2) / (1 + REDSHIFT)
    maggies_unc = (unc_jy / 3631.0) * (4 * np.pi * d_lum**2) / (1 + REDSHIFT)

    # --- 4. Create a Mask ---
    # Mask out regions you don't want to fit (e.g., noisy edges).
    mask = (wave_rest > 3700) & (wave_rest < 7000)
    print(f"Data prepared. Using wavelength range {wave_rest[mask].min():.0f}-{wave_rest[mask].max():.0f} A.")

    # =========================================================================
    # Step 3: Build the Prospector `obs` Dictionary
    # =========================================================================
    print("\n--- Step 3: Building Prospector 'obs' dictionary ---")
    
    obs = {}
    obs['redshift'] = REDSHIFT
    obs['wavelength'] = wave_rest
    obs['spectrum'] = maggies
    obs['unc'] = maggies_unc
    obs['mask'] = mask
    obs['phot_wave'] = [] # No photometry in this example
    obs['maggies'] = np.array([])
    obs['maggies_unc'] = np.array([])
    print("'obs' dictionary created successfully.")

    # =========================================================================
    # Step 4: Build the Prospector `model`
    # =========================================================================
    print("\n--- Step 4: Building Prospector 'model' ---")
    
    from prospector.models.templates import TemplateLibrary
    from prospector.models import SpecModel

    model_params = TemplateLibrary["parametric_sfh"]
    model_params.update(TemplateLibrary["dust_emission"])

    model_params["logmass"]["prior"] = {"name": "uniform", "mini": 8.0, "maxi": 12.0}
    model_params["logzsol"]["prior"] = {"name": "uniform", "mini": -1.0, "maxi": 0.19}
    
    model = SpecModel(model_params)
    print("Model created with a delayed-tau SFH and dust emission.")

    # =========================================================================
    # Step 5: Run the Fit
    # =========================================================================
    print("\n--- Step 5: Running the fit (this may take a long time!) ---")
    
    import prospector.fitting as fitting
    from prospector.io import write_results as writer

    sps = fitting.get_sps(zcontinuous=1)

    # We use dynesty for nested sampling.
    # For a real run, you might increase nlive_init and nlive_batch.
    output = fitting.fit_model(obs, model, sps,
                               fitting_method='dynesty',
                               noise_model='gp_spec',
                               dynesty_kwargs={'nlive_init': 200, 'nlive_batch': 200,
                                               'maxbatch': 10, 'sample': 'rwalk'})
    
    print("Fit complete.")
    writer.write_hdf5(RESULTS_FILE, {}, model, obs, output['sampling'][0],
                      output['optimization'][0], tsample=output['sampling'][1],
                      toptimize=output['optimization'][1])
    print(f"Results saved to '{RESULTS_FILE}'")

def analyze_results():
    """
    Loads the results from the HDF5 file and prints the stellar mass.
    """
    print("\n--- Step 6: Analyzing the results ---")
    
    import prospector.io.read_results as reader

    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Results file not found at '{RESULTS_FILE}'")
        return

    results_type, samps, model_out = reader.results_from(RESULTS_FILE)

    # Extract the posterior for the logmass parameter
    logmass_posterior = samps['chain'][:, samps['theta_labels'].index('logmass')]

    # Calculate the 16th, 50th, and 84th percentiles
    mass_percentiles = np.percentile(logmass_posterior, [16, 50, 84])
    log_mass_median = mass_percentiles[1]
    log_mass_lower_err = mass_percentiles[1] - mass_percentiles[0]
    log_mass_upper_err = mass_percentiles[2] - mass_percentiles[1]

    print("\n--- Stellar Mass Results ---")
    print(f"Stellar Mass (log M_sun): {log_mass_median:.2f} (+{log_mass_upper_err:.2f} / -{log_mass_lower_err:.2f})")
    
    # Optional: Create and save a corner plot to visualize all posteriors
    try:
        from prospect.utils.plotting import corner_plot
        print("\nGenerating corner plot...")
        fig = corner_plot(samps)
        fig.savefig("corner_plot.png")
        print("Corner plot saved to 'corner_plot.png'")
    except ImportError:
        print("\nCould not generate corner plot: prospect.utils.plotting not found.")
        print("This is an optional dependency.")


if __name__ == '__main__':
    # This script can be run in two stages:
    # 1. Run the fit: python SED_tutorial.py --fit
    # 2. Analyze the results: python SED_tutorial.py --analyze
    
    import argparse
    parser = argparse.ArgumentParser(description="Run or analyze a Prospector SED fit.")
    parser.add_argument('--fit', action='store_true', help='Run the fitting process.')
    parser.add_argument('--analyze', action='store_true', help='Analyze the results from a previous fit.')
    args = parser.parse_args()

    if args.fit:
        # Check for required libraries before starting
        try:
            import mpdaf
            import prospector
            import fsps
            import dynesty
        except ImportError as e:
            print(f"Error: Missing required library. Please install it. ({e})")
        else:
            run_prospector_fit()
            
    elif args.analyze:
        analyze_results()
        
    else:
        print("Please specify an action: --fit or --analyze")
        print("Example: python SED_tutorial.py --fit")
