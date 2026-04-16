
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits

try:
    tab = Table(fits.open('coadd_spectra.fits')[1].data)
except FileNotFoundError:
    print("Error: 'coadd_spectra.fits' not found.")
    print("Please ensure the FITS file is in the same directory as this script.")
    exit()

rail = np.arange(620, 9500, 2)
grid = np.full((len(rail), len(tab)), np.nan)

for i, t in enumerate(tab):
    if len(t['Wavelength']) == 0:
        continue
        
    start_idx = np.searchsorted(rail, t['Wavelength'][0])
    
    end_idx = start_idx + len(t['Wavelength'])
    if end_idx <= len(rail):
        grid[start_idx:end_idx, i] = t['Flux']

coadd = np.nanmean(grid, axis=-1)
indicator = np.sum(~np.isnan(grid), axis=-1)

# Normalize the coadded spectrum so its median is 1
median_flux = np.nanmedian(coadd)
if median_flux > 0:
    coadd /= median_flux

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharey=True)

wave_splits = np.linspace(rail.min(), rail.max(), 4)
max_indicator = np.nanmax(indicator)

for i, ax in enumerate(axes):
    start_wave, end_wave = wave_splits[i], wave_splits[i+1]
    
    idx = (rail >= start_wave) & (rail < end_wave)
    rail_slice = rail[idx]
    coadd_slice = coadd[idx]
    indicator_slice = indicator[idx]

    ax.plot(rail_slice, coadd_slice, color='dodgerblue', lw=1)
    ax.fill_between(rail_slice, coadd_slice, 0, color='lightblue', alpha=0.5)
    ax.set_ylabel("Normalized Flux")
    ax.set_xlim(start_wave, end_wave)
    ax.grid(False)

    if i == len(axes) - 1:
        ax.set_xlabel("Wavelength (Å)")

    ax2 = ax.twinx()
    
    ax2.fill_between(rail_slice, indicator_slice, color='gray', alpha=0.3, step='pre')
    
    ax2.set_ylim(0, max_indicator * 3)
    ax2.set_ylabel("Coverage")

fig.tight_layout()
plt.show()
