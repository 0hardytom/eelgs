import pandas as pd
import numpy as np
from mpdaf.obj import Cube, Spectrum
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import os
from tqdm import tqdm
from mpi4py import MPI

def process_spectrum(row, cube):
    """
    Processes a single spectrum from a FITS cube based on a row of metadata.
    """
    try:
        # Create a circular aperture and extract the spectrum
        spec = cube.aperture((row['Y_PEAK_SN'], row['X_PEAK_SN']), 2, unit_center=None)

        # Go to rest frame by creating a new Spectrum object
        z = row['Redshift']
        observed_wave = spec.wave.coord()
        rest_wave = observed_wave / (1 + z)
        rest_spec = Spectrum(wave=rest_wave, data=spec.data, var=spec.var)

        # Manually calculate the continuum using a 20 Angstrom rolling median
        wave = rest_spec.wave.coord()
        flux = rest_spec.data
        continuum_flux = np.zeros_like(flux)
        half_window = 10.0  # 20 Angstrom window -> 10 on each side

        for i in range(len(wave)):
            w_center = wave[i]
            start_idx = np.searchsorted(wave, w_center - half_window, side='left')
            end_idx = np.searchsorted(wave, w_center + half_window, side='right')
            
            window_flux = flux[start_idx:end_idx]
            if window_flux.size > 0:
                continuum_flux[i] = np.median(window_flux)
            else:
                continuum_flux[i] = flux[i]
        
        continuum_flux[continuum_flux < 1e-6] = 1e-6
        normalized_flux = rest_spec.data / continuum_flux

        # Manually resample to a 2 Angstrom grid
        wave_min = np.ceil(rest_spec.wave.get_start() / 2.) * 2.
        wave_max = np.floor(rest_spec.wave.get_end() / 2.) * 2.
        
        if wave_min >= wave_max:
            return None, None
            
        resampled_wave = np.arange(wave_min, wave_max + 2.0, 2.0)
        interp_func = interp1d(rest_spec.wave.coord(), normalized_flux, kind='linear', bounds_error=False, fill_value=np.nan)
        resampled_flux = interp_func(resampled_wave)
        
        valid_indices = ~np.isnan(resampled_flux)
        return resampled_wave[valid_indices], resampled_flux[valid_indices]

    except Exception as e:
        # Suppress verbose error printing in MPI mode unless it's rank 0
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Could not process ID {row['ID']}: {e}")
        return None, None

def main():
    """
    Main function to run the co-addition preparation script, with MPI support.
    """
    # MPI setup
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    size = COMM.Get_size()

    # Define paths
    csv_path = 'leadlines.csv'
    base_fits_path = '/Volumes/Expansion/exp_thardy/'
    output_fits_path = 'coadd_spectra.fits'

    if rank == 0:
        print(f"Running with {size} MPI processes.")
        print(f"Reading metadata from {csv_path}")
    
    # All processes read the CSV to have the metadata
    try:
        df = pd.read_csv(csv_path)
        df = df[df['Redshift'] > 0]
    except FileNotFoundError:
        if rank == 0:
            print(f"Error: {csv_path} not found.")
        return

    # Group by cube file to create a list of jobs
    grouped = df.groupby(['dir', 'key'])
    jobs = list(grouped.groups.keys())

    # Distribute jobs among MPI processes
    jobs_for_this_rank = jobs[rank::size]

    local_spectra = []
    
    # Use tqdm only for rank 0 to avoid messy output
    if rank == 0:
        pbar = tqdm(total=len(jobs), desc="Processing Cubes")
    
    for i, (directory, key) in enumerate(jobs_for_this_rank):
        fits_path = os.path.join(base_fits_path, directory, f"{key}_COMBINED_CUBE_MED_FINAL.fits")
        
        if not os.path.exists(fits_path):
            continue
            
        try:
            cube = Cube(fits_path)
            # Get all rows for the current cube
            group_df = grouped.get_group((directory, key))
            for _, row in group_df.iterrows():
                wavelength, flux = process_spectrum(row, cube)
                if wavelength is not None and flux is not None:
                    local_spectra.append({
                        'ID': row['ID'],
                        'Redshift': row['Redshift'],
                        'Wavelength': wavelength,
                        'Flux': flux
                    })
        except Exception as e:
            if rank == 0:
                print(f"Could not process cube {fits_path}: {e}")
        
        # Manually update progress for rank 0
        if rank == 0:
            # This is an approximation of progress, as work is distributed
            pbar.update(size if i > 0 else 1)

    if rank == 0:
        pbar.close()

    # Gather all results to the root process
    all_spectra_lists = COMM.gather(local_spectra, root=0)

    if rank == 0:
        if not all_spectra_lists:
            print("No spectra were processed. Exiting.")
            return

        # Flatten the list of lists
        all_spectra = [item for sublist in all_spectra_lists for item in sublist]

        if not all_spectra:
            print("No spectra were successfully processed after gathering. Exiting.")
            return

        # Create and save the final FITS file
        table = Table(all_spectra)
        print(f"Saving master table with {len(all_spectra)} entries to {output_fits_path}")
        hdu = fits.BinTableHDU(table)
        hdu.writeto(output_fits_path, overwrite=True)
        print("Done.")

if __name__ == '__main__':
    main()
