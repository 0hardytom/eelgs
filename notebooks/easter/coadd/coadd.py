
import pandas as pd
import numpy as np
from mpdaf.obj import Cube, Spectrum
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import os
from tqdm import tqdm

def process_spectrum(row, cube):
    """
    Processes a single spectrum from a FITS cube based on a row of metadata.
    """
    # try:
    # Create a circular aperture and extract the spectrum
    print(f'processing {row['I']},{row['ID']}')
    spec = cube.aperture((row['Y_PEAK_SN'], row['X_PEAK_SN']), 2, unit_center=None)

    # Go to rest frame by creating a new Spectrum object
    z = row['Redshift']
    # rest_wave_coord = spec.wave.restframe(z)
    observed_wave = spec.wave.coord()
    rest_wave = observed_wave / (1 + z)
    # rest_spec = Spectrum(wave=rest_wave, data=spec.data, var=spec.var)

    # Manually calculate the continuum using a 20 Angstrom rolling median
    # wave = rest_wave
    flux = spec.data
    continuum_flux = np.zeros_like(flux)
    # half_window = 10.0  # 20 Angstrom window -> 10 on each side

    # for i in tqdm(range(len(wave))):
    #     w_center = wave[i]
    #     # Find start and end indices for the window using a fast search
    #     start_idx = np.searchsorted(wave, w_center - half_window, side='left')
    #     end_idx = np.searchsorted(wave, w_center + half_window, side='right')
        
    #     window_flux = flux[start_idx:end_idx]
    #     if window_flux.size > 0:
    #         continuum_flux[i] = np.median(window_flux)
    #     else:
    #         continuum_flux[i] = flux[i]  # Fallback if window is empty
    
    flux_series = pd.Series(flux)
    continuum_flux = flux_series.rolling(window=26, center=True,min_periods=1).median().values


    # Avoid division by zero or near-zero
    continuum_flux[continuum_flux < 1e-6] = 1e-6
    normalized_flux = spec.data / continuum_flux

    # Manually resample to a 2 Angstrom grid using interpolation
    wave_min = np.ceil(rest_wave[0] / 2.) * 2.
    wave_max = np.floor(rest_wave[-1] / 2.) * 2.
    
    if wave_min >= wave_max:
        print(f"Wavelength range too small for resampling for ID {row['ID']}")
        return None, None
        
    # Define the new, evenly spaced wavelength grid
    resampled_wave = np.arange(wave_min, wave_max + 2.0, 2.0)
    
    # Create an interpolation function and apply it to the new grid
    interp_func = interp1d(rest_wave, normalized_flux, kind='linear', bounds_error=False, fill_value=np.nan)
    resampled_flux = interp_func(resampled_wave)
    
    # Filter out any NaN values that may have resulted from interpolation
    valid_indices = ~np.isnan(resampled_flux)
    
    return resampled_wave[valid_indices], resampled_flux[valid_indices]

    # except Exception as e:
    #     print(f"Could not process ID {row['ID']}: {e}")
    #     return None, None

def main():
    """
    Main function to run the co-addition preparation script.
    """
    # Define paths
    csv_path = 'leadlines.csv'
    base_fits_path = '/Volumes/Expansion/exp_thardy/'
    output_fits_path = 'coadd_spectra.fits'

    # Read the CSV
    print(f"Reading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter out bad redshift values
    df = df[df['Redshift'] > 0]

    all_spectra = []
    
    print("Processing spectra...")
    
    # Group by cube file to avoid reopening files
    grouped = df.groupby(['dir', 'key'])
    
    for (directory, key), group in tqdm(grouped, total=len(grouped)):
        fits_path = os.path.join(base_fits_path, directory, f"{key}_COMBINED_CUBE_MED_FINAL.fits")
        
        if not os.path.exists(fits_path):
            print(f"File not found: {fits_path}")
            continue
            
        # try:
        cube = Cube(fits_path)
        for _, row in group.iterrows():
            wavelength, flux = process_spectrum(row, cube)
            if wavelength is not None and flux is not None:
                # Store as a dictionary
                all_spectra.append({
                    'ID': row['ID'],
                    'Redshift': row['Redshift'],
                    'Wavelength': wavelength,
                    'Flux': flux
                })
        # except Exception as e:
            # print(f"Could not process cube {fits_path}: {e}")


    if not all_spectra:
        print("No spectra were successfully processed. Exiting.")
        return

    # Create an Astropy Table
    # The arrays can be of different lengths, so we store them as objects
    table = Table(all_spectra)

    # Save to a FITS file
    print(f"Saving master table to {output_fits_path}")
    hdu = fits.BinTableHDU(table)
    hdu.writeto(output_fits_path, overwrite=True)
    print("Done.")

if __name__ == '__main__':
    main()
