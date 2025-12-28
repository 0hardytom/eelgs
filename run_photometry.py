import os
import shutil
import pandas as pd
import numpy as np
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.table import Table

from astroquery.mast import Observations
from astroquery.irsa import Irsa

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import LocalBackground

# --- Configuration ---
INPUT_CSV = 'notebooks/xmas/catalogue_analysis/peas_test.csv'
OUTPUT_CSV = 'photometry_results.csv'
DATA_DIR = 'phot_data'
APERTURE_RADIUS_ARCSEC = 2.0
SKY_ANNULUS_INNER_ARCSEC = 4.0
SKY_ANNULUS_OUTER_ARCSEC = 6.0

# Band definitions: (query_name, instrument, archive)
BANDS = {
    # HST - Using common filters for ACS and WFC3
    'HST_F275W': ('F275W', 'UVIS', 'MAST'),
    'HST_F435W': ('F435W', 'ACS', 'MAST'),
    'HST_F606W': ('F606W', 'ACS', 'MAST'),
    'HST_F814W': ('F814W', 'ACS', 'MAST'),
    'HST_F125W': ('F125W', 'IR', 'MAST'),
    'HST_F160W': ('F160W', 'IR', 'MAST'),
    # Spitzer - Using channel names for IRAC and MIPS
    'Spitzer_I1_3.6': ('1', 'IRAC', 'Spitzer'),
    'Spitzer_I2_4.5': ('2', 'IRAC', 'Spitzer'),
    'Spitzer_I4_8.0': ('4', 'IRAC', 'Spitzer'),
    'Spitzer_M1_24': ('1', 'MIPS', 'Spitzer'),
}

# Suppress verbose warnings for a cleaner output
warnings.simplefilter('ignore', category=AstropyWarning)

def download_hst_image(coord, band_info, download_dir):
    """Queries MAST and downloads the first available HST image for a given band."""
    print(f"  Querying MAST for {band_info[0]}...")
    try:
        obs_table = Observations.query_criteria(
            obs_collection="HST",
            instrument_name=f"WFC3/{band_info[1]}" if "UVIS" in band_info[1] or "IR" in band_info[1] else f"{band_info[1]}/WFC",
            filters=band_info[0],
            coordinates=coord,
            t_exptime=[100, 99999], # Avoid very short exposures
            dataproduct_type="image",
        )
        
        if not obs_table:
            print(f"  No suitable HST observations found for {band_info[0]}.")
            return None

        # Prioritize DRZ/DRC files (drizzled, corrected)
        products = Observations.get_product_list(obs_table)
        science_products = Observations.filter_products(products,
            productSubGroupDescription=['DRZ', 'DRC'],
            extension="fits"
        )

        if not science_products:
             print(f"  No DRZ/DRC science products found for {band_info[0]}. Trying any FITS.")
             science_products = Observations.filter_products(products, extension="fits")

        if not science_products:
            print(f"  No FITS products found for {band_info[0]}.")
            return None

        # Download the first result
        Observations.download_products(science_products[0], download_dir=download_dir, mrp_only=False)
        
        # Find the downloaded file path
        for root, _, files in os.walk(download_dir):
            for file in files:
                if file.endswith((".fits", ".fits.gz")):
                    return os.path.join(root, file)
        return None

    except Exception as e:
        print(f"  Error downloading HST data for {band_info[0]}: {e}")
        return None


def download_spitzer_image(coord, band_info, download_dir):
    """Queries IRSA and downloads the first available Spitzer image."""
    instrument = band_info[1]
    channel = band_info[0]
    print(f"  Querying IRSA for Spitzer/{instrument} Ch{channel}...")
    try:
        # Search within the Spitzer Heritage Archive (SHA)
        table = Irsa.query_region(coord, catalog='spitzer_sha', spatial='Cone', radius=5 * u.arcmin)
        
        # Filter for the correct instrument, channel, and data type (Level 2 mosaic)
        mask = (
            (table['instrument'] == instrument) &
            (table['ch'] == int(channel)) &
            (table['prodtype'] == 'pbcd') # Post-Basic Calibrated Data (Level 2)
        )
        filtered_table = table[mask]

        if not filtered_table:
            print(f"  No suitable Spitzer/{instrument} Ch{channel} observations found.")
            return None

        # Get the download URL for the mosaic FITS file
        url = filtered_table[0]['accessUrl']
        download_path = os.path.join(download_dir, f"spitzer_{instrument}_{channel}.fits")
        Irsa.download_file(url, download_path)
        return download_path

    except Exception as e:
        print(f"  Error downloading Spitzer data for {instrument} Ch{channel}: {e}")
        return None

def perform_photometry(image_path, coord):
    """Performs aperture photometry on a single FITS image."""
    print(f"  Performing photometry on {os.path.basename(image_path)}...")
    try:
        with fits.open(image_path, memmap=False) as hdul:
            # Find the science data extension
            sci_ext = 0
            for i, hdu in enumerate(hdul):
                if hdu.header.get('EXTNAME') == 'SCI':
                    sci_ext = i
                    break
            
            wcs = WCS(hdul[sci_ext].header)
            data = hdul[sci_ext].data

            # Convert RA/Dec to pixel coordinates
            px, py = wcs.world_to_pixel(coord)
            position = (px, py)

            # Define apertures
            aperture = CircularAperture(position, r=APERTURE_RADIUS_ARCSEC / wcs.proj_plane_pixel_scales()[0].to(u.arcsec / u.pix).value)
            annulus = CircularAnnulus(position, 
                                      r_in=SKY_ANNULUS_INNER_ARCSEC / wcs.proj_plane_pixel_scales()[0].to(u.arcsec / u.pix).value,
                                      r_out=SKY_ANNULUS_OUTER_ARCSEC / wcs.proj_plane_pixel_scales()[0].to(u.arcsec / u.pix).value)
            
            # Perform background-subtracted photometry
            bkg_phot = aperture_photometry(data - LocalBackground(annulus, data).background, aperture, wcs=wcs)
            
            # Unit conversion to microjanskys (uJy)
            flux_ujy = -999.0
            if 'PHOTFLAM' in hdul[sci_ext].header: # HST ABmag systems
                photflam = hdul[sci_ext].header['PHOTFLAM'] * u.erg / u.s / u.cm**2 / u.AA
                photplam = hdul[sci_ext].header['PHOTPLAM'] * u.AA
                flux_density = (bkg_phot['aperture_sum'][0] * photflam).to(u.uJy, u.spectral_density(photplam))
                flux_ujy = flux_density.value
            elif 'FLUXMJY' in hdul[sci_ext].header: # Spitzer (MJy/sr)
                flux_per_pixel_mjy = hdul[sci_ext].header['FLUXMJY']
                pixel_area_sr = wcs.proj_plane_pixel_area().to(u.sr).value
                flux_density_mjy = bkg_phot['aperture_sum'][0] * flux_per_pixel_mjy * pixel_area_sr
                flux_ujy = (flux_density_mjy * u.MJy).to(u.uJy).value

            print(f"  Flux: {flux_ujy:.2f} uJy")
            return flux_ujy

    except Exception as e:
        print(f"  Could not perform photometry: {e}")
        return -999.0 # Return a sentinel value for failure

def cleanup_dir_contents(directory):
    """Deletes all files and subdirectories within a given directory."""
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f'  Warning: Failed to delete {item_path}. Reason: {e}')

def main():
    """Main script execution."""
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at {INPUT_CSV}")
        return

    # Prepare directories and output file
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    df_in = pd.read_csv(INPUT_CSV)
    results = []

    print(f"Starting photometry for {len(df_in)} galaxies...")

    for index, row in df_in.iterrows():
        galaxy_id = row['ID']
        coord = SkyCoord(row['RA'], row['DEC'], unit=(u.deg, u.deg))
        print(f"\nProcessing Galaxy: {galaxy_id} ({coord.to_string('hmsdms')})")

        galaxy_photometry = {'ID': galaxy_id, 'RA': row['RA'], 'DEC': row['DEC']}
        galaxy_data_dir = os.path.join(DATA_DIR, str(galaxy_id))
        os.makedirs(galaxy_data_dir, exist_ok=True)

        for band_name, band_info in BANDS.items():
            query_name, instrument, archive = band_info
            
            image_path = None
            if archive == 'MAST':
                image_path = download_hst_image(coord, (query_name, instrument), galaxy_data_dir)
            elif archive == 'Spitzer':
                image_path = download_spitzer_image(coord, (query_name, instrument), galaxy_data_dir)

            if image_path and os.path.exists(image_path):
                flux = perform_photometry(image_path, coord)
                galaxy_photometry[f'flux_{band_name}'] = flux
                # Clean up the downloaded data immediately and robustly
                cleanup_dir_contents(galaxy_data_dir)
                print(f"  Cleaned up image data directory for {band_name}.")
            else:
                galaxy_photometry[f'flux_{band_name}'] = -999.0
        
        results.append(galaxy_photometry)
        # Save progress incrementally
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Final save
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPhotometry complete. Results saved to {OUTPUT_CSV}")
    
    # Final cleanup
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"Cleaned up temporary data directory: {DATA_DIR}")

if __name__ == "__main__":
    main()