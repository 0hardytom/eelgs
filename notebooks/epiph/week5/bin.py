import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo # Using a standard cosmology
import astropy.units as u

# --- Your Data and Survey Parameters ---

# This is a placeholder for your redshift data.
# REPLACE this with your actual array of galaxy redshifts.
z = np.random.uniform(0.3, 0.8, size=500)

# Survey area in square arcminutes
survey_area_sq_arcmin = 38.0

# Redshift range and binning
z_min = 0.3
z_max = 0.8
z_bins = np.linspace(z_min, z_max, 11) # 10 bins from 0.3 to 0.8

# --- Calculations ---

# 1. Convert survey area to steradians
survey_area = survey_area_sq_arcmin * (u.arcmin**2)
survey_area_sr = survey_area.to(u.sr)

# 2. Get the number of galaxies in each redshift bin
galaxy_counts, _ = np.histogram(z, bins=z_bins)

# 3. Calculate the comoving volume for each bin
bin_volumes = []
for i in range(len(z_bins) - 1):
    z_low = z_bins[i]
    z_high = z_bins[i+1]
    
    # Calculate the comoving volume of the shell for the full sky
    v_low = cosmo.comoving_volume(z_low)
    v_high = cosmo.comoving_volume(z_high)
    shell_volume = v_high - v_low
    
    # Scale the volume to your survey area
    # The full sky is 4*pi steradians
    volume_in_bin = shell_volume * (survey_area_sr.value / (4 * np.pi))
    bin_volumes.append(volume_in_bin.to(u.Mpc**3).value)

bin_volumes = np.array(bin_volumes)

# 4. Calculate the volume density (number of galaxies per cubic Megaparsec)
# Avoid division by zero if a bin has zero volume (though unlikely here)
volume_density = np.divide(galaxy_counts, bin_volumes, out=np.zeros_like(galaxy_counts, dtype=float), where=bin_volumes!=0)

# Get the center of each redshift bin for plotting
z_bin_centers = (z_bins[:-1] + z_bins[1:]) / 2

# --- Plotting ---

plt.figure(figsize=(10, 6))
# Using a step plot is common for binned data
plt.step(z_bin_centers, volume_density, where='mid')

plt.xlabel('Redshift (z)')
plt.ylabel('Volume Density (N / Mpc$^3$)')
plt.title('Galaxy Volume Density vs. Redshift')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# You can also print the results
print("Redshift Bin Centers | Galaxy Count | Bin Volume (Mpc^3) | Volume Density (N/Mpc^3)")
print("------------------------------------------------------------------------------------")
for i in range(len(z_bin_centers)):
    print(f"{z_bin_centers[i]:<20.3f} | {galaxy_counts[i]:<12d} | {bin_volumes[i]:<20.2e} | {volume_density[i]:.2e}")
