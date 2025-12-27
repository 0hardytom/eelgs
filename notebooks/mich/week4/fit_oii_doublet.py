

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

# --- 1. Define the physical properties of the [OII] doublet ---
WAVE_1 = 3726.03  # Wavelength of the first line (rest frame)
WAVE_2 = 3728.81  # Wavelength of the second line (rest frame)
WAVE_SEPARATION = WAVE_2 - WAVE_1

# --- 2. Create some realistic-looking fake data ---
# Let's assume the doublet is at a small redshift z=0.05
z = 0.05
obs_wave_1 = WAVE_1 * (1 + z)
obs_wave_2 = WAVE_2 * (1 + z)

# Wavelength axis for our fake spectrum
wavelength = np.linspace(3890, 3940, 200)

# Create the two Gaussian components
# Assume the second line is 1.5x brighter than the first
flux_amp_1 = 10
flux_amp_2 = 15
linewidth_sigma = 1.5 # In Angstroms

g1 = models.Gaussian1D(amplitude=flux_amp_1, mean=obs_wave_1, stddev=linewidth_sigma)
g2 = models.Gaussian1D(amplitude=flux_amp_2, mean=obs_wave_2, stddev=linewidth_sigma)

# Add them together and add some noise
true_flux = g1(wavelength) + g2(wavelength)
noise = np.random.normal(0., 1.0, wavelength.shape)
observed_flux = true_flux + noise

# --- 3. Set up the composite model for fitting ---
# Initial guesses for the fitter
amp_guess = 12
mean_guess = obs_wave_1 + 1 # Start slightly off
sigma_guess = 1.2

# Create two Gaussian models for the fit
g1_init = models.Gaussian1D(amplitude=amp_guess, mean=mean_guess, stddev=sigma_guess,
                           bounds={'amplitude': (0, None), 'stddev': (0.5, None)})
g2_init = models.Gaussian1D(amplitude=amp_guess, mean=mean_guess + WAVE_SEPARATION, stddev=sigma_guess,
                           bounds={'amplitude': (0, None)})

# This is the key part: Tie parameters together based on physics
def tie_mean(model):
    """Tie the mean of the second Gaussian to the first."""
    # The separation in the observed frame is scaled by (1+z)
    # A more robust way is to fit for z, but for simplicity we use a fixed z here.
    obs_separation = WAVE_SEPARATION * (1 + z)
    return model.mean_0 + obs_separation

def tie_stddev(model):
    """Tie the stddev of the second Gaussian to the first."""
    return model.stddev_0

# Create the composite model
oii_doublet_init = g1_init + g2_init

# Apply the constraints
oii_doublet_init.mean_1.tied = tie_mean
oii_doublet_init.stddev_1.tied = tie_stddev


# --- 4. Run the fitter ---
fitter = fitting.LevMarLSQFitter()
# We add weights to the fitter to ignore the noisiest points if needed, here we use uniform weights
# weights = 1.0 / np.sqrt(np.abs(observed_flux)) # Example of weighting
fitted_oii_model = fitter(oii_doublet_init, wavelength, observed_flux)

# --- 5. Calculate the flux and display results ---
# Get the fitted components
fitted_g1 = fitted_oii_model[0]
fitted_g2 = fitted_oii_model[1]

# The flux is the integral of the Gaussian: A * sigma * sqrt(2*pi)
flux_1 = fitted_g1.amplitude.value * fitted_g1.stddev.value * np.sqrt(2 * np.pi)
flux_2 = fitted_g2.amplitude.value * fitted_g2.stddev.value * np.sqrt(2 * np.pi)
total_flux = flux_1 + flux_2

print("--- Fit Results ---")
print(f"Fitted Amplitude 1: {fitted_g1.amplitude.value:.2f}")
print(f"Fitted Amplitude 2: {fitted_g2.amplitude.value:.2f}")
print(f"Fitted Mean Wavelength 1: {fitted_g1.mean.value:.2f} Å")
print(f"Fitted Mean Wavelength 2: {fitted_g2.mean.value:.2f} Å")
print(f"Fitted Linewidth (sigma): {fitted_g1.stddev.value:.2f} Å")
print("\n--- Flux Calculation ---")
print(f"Flux of component 1: {flux_1:.2f}")
print(f"Flux of component 2: {flux_2:.2f}")
print(f"TOTAL FLUX of [OII] doublet: {total_flux:.2f}")


# --- 6. Plot the results for visual confirmation ---
plt.figure(figsize=(10, 6))
plt.plot(wavelength, observed_flux, 'ko', label='Observed Data', markersize=4)
plt.plot(wavelength, fitted_oii_model(wavelength), 'r-', lw=2, label='Total Fit')
plt.plot(wavelength, fitted_g1(wavelength), 'b--', label='Component 1')
plt.plot(wavelength, fitted_g2(wavelength), 'g--', label='Component 2')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux')
plt.title('[OII] Doublet Fit')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()
