import numpy as np
import matplotlib.pyplot as plt
from prospect.io import read_results as pread

# --- This section should be edited by the user ---
# Specify the path to your Prospector results file
FILE = "your_results_file.h5" 
# --- End of user-editable section ---

try:
    # Load the results from the HDF5 file
    results, observations_dict, _ = pread.results_from(FILE)
except FileNotFoundError:
    print(f"Error: The file '{FILE}' was not found.")
    print("Please update the 'FILE' variable in this script with the correct path.")
    exit()


# --- How to reconstruct the best-fit parameters ---

# The MCMC chain contains all the parameter samples.
# Shape is typically (nwalkers, nsteps, nparams)
chain = results.get('chain', None)
lnprobability = results.get('lnprobability', None)

if chain is None or lnprobability is None:
    print("Error: 'chain' or 'lnprobability' not found in the results file.")
    exit()

# To find the MAP, we find the flat index of the maximum lnprobability
# We need to reshape the chain and lnprobability arrays to be flat
flat_lnprobability = lnprobability.reshape(-1)
flat_chain = chain.reshape(-1, chain.shape[-1])

# Find the index of the highest probability
best_fit_index = np.argmax(flat_lnprobability)

# Use that index to get the best-fit parameter vector
bestfit_params = flat_chain[best_fit_index]

# --- Now you can proceed with the code from before ---

# 1. Get the model and sps object from the results
model = results.get('model', None)
sps = results.get('sps', None)

if model is None or sps is None:
    print("Error: 'model' or 'sps' object not found in the results file.")
    exit()

# 2. Generate the best-fit spectrum and photometry
spec, phot, mfrac = model.predict(bestfit_params, obs=observations_dict, sps=sps)

# 3. Prepare for plotting
obs_phot = observations_dict.get('photometry', None)
obs_phot_unc = observations_dict.get('photometric_uncertainties', None)

if obs_phot is None or obs_phot_unc is None:
    print("Error: Photometry data not found in the observations dictionary.")
    exit()

filter_wavelengths = np.array([f.wave_effective for f in observations_dict['filters']])
wave_spec = sps.wavelengths

# 4. Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot observed and model photometry, and the best-fit spectrum
ax.errorbar(filter_wavelengths, obs_phot, yerr=obs_phot_unc,
            marker='o', markersize=8, linestyle='', color='k',
            label='Observed Photometry', zorder=10)
ax.plot(filter_wavelengths, phot, marker='s', markersize=10, linestyle='',
        color='red', label='Model Photometry (Best-fit)')
ax.plot(wave_spec, spec, color='cornflowerblue', alpha=0.8,
        label='Best-fit Spectrum')

# --- Finalize the plot ---
ax.set_xlabel('Wavelength (Angstroms)', fontsize=14)
ax.set_ylabel('Flux (Maggies)', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=12)
ax.set_title('Posterior Predictive Check (Reconstructed Best-fit)', fontsize=16)

# Save the figure to a file
output_filename = "posterior_predictive_check.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")

plt.show()
