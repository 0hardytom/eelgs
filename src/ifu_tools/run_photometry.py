import os
import shutil
import pandas as pd
import numpy as np
import warnings
import tarfile

import pyvo as vo
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales, proj_plane_pixel_area
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.table import Table

from astroquery.mast import Observations
from astroquery.ipac.irsa import Irsa
from astroquery.heasarc import Heasarc
# from astroquery.ukssdc import Ukssdc
# import swifttools.ukssdc as Ukssdc
from astroquery.esa.xmm_newton import XMMNewton
from astroquery.vizier import Vizier
from astropy.utils.data import download_file
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import LocalBackground
from regions import PolygonSkyRegion, CircleSkyRegion
from regions.core import PixCoord

# --- Configuration ---
INPUT_CSV = 'allsourcesNOSTACK.csv'
OUTPUT_CSV = 'photometry_results.csv'
DATA_DIR = 'phot-temp'
APERTURE_RADIUS_ARCSEC = 2.0
SKY_ANNULUS_INNER_ARCSEC = 4.0
SKY_ANNULUS_OUTER_ARCSEC = 6.0

# Band definitions: (query_name, instrument, archive)
BANDS = {
    # HST - Using common filters for ACS and WFC3
    # 'HST_F218W': ('F218W', 'UVIS', 'MAST'),
    # 'HST_F225W': ('F225W', 'UVIS', 'MAST'),
    # 'HST_F275W': ('F275W', 'UVIS', 'MAST'),
    'HST_F435W': ('F435W', 'ACS', 'MAST'),
    'HST_F606W': ('F606W', 'ACS', 'MAST'),
    'HST_F814W': ('F814W', 'ACS', 'MAST'),
    'HST_F125W': ('F125W', 'IR', 'MAST'),
    'HST_F160W': ('F160W', 'IR', 'MAST'),
    # Spitzer - Using channel names for IRAC and MIPS
    'Spitzer_I1_3.6': ('1', 'IRAC', 'Spitzer'),
    'Spitzer_I2_4.5': ('2', 'IRAC', 'Spitzer'),
    'Spitzer_M1_24': ('1', 'MIPS', 'Spitzer'),
    'Spitzer_M2_70': ('2', 'MIPS', 'Spitzer')}

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
        
        if 'IR' in band_info[1]:
            polymask = np.array(['POLYGON' in item[7:] for item in obs_table['s_region'] ])
            obs_table_clean = obs_table[polymask]
        else:
            obs_table_clean = obs_table

        products = Observations.get_product_list(obs_table_clean)
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
        
        Observations.download_products(science_products[0], download_dir=download_dir, mrp_only=False)
        
        for root, _, files in os.walk(download_dir):
            for file in files:
                if file.endswith((".fits", ".fits.gz")):
                    return os.path.join(root, file)
        return None

    except Exception as e:
        print(f"  Error downloading HST data for {band_info[0]}: {e}")
        return None


def download_spitzer_image(coord, band_info, download_dir):
    """Queries IRSA using SIA and downloads the first available Spitzer image."""
    seip_service2= vo.dal.sia2.SIA2Service('https://irsa.ipac.caltech.edu/SIA')
    instrument = band_info[1]
    channel = band_info[0]
    print(f"  Querying IRSA for Spitzer/{instrument} Ch{channel}...")
    try:
        radius_deg = (5 * u.arcmin).to(u.deg).value
        im_table = seip_service2.search(pos=(coord.ra.deg, coord.dec.deg, radius_deg),
                                collection='spitzer_sha')

        if not im_table:
            print(f"  No suitable Spitzer/{instrument} Ch{channel} observations found.")
            return None
        
        table = im_table.to_table()
        mask = (table['instrument_name'] == instrument
                )&(table['energy_bandpassname'] == instrument+channel
                   )&(table['calib_level'] == 2
                      )&(table['dataproduct_subtype']=='science')
        filtered_table = table[mask]

        if not filtered_table:
            print(f"  No suitable Level 2 Spitzer/{instrument} Ch{channel} observations found.")
            return None

        url = filtered_table[0]['access_url']
        download_path = os.path.join(download_dir, f"spitzer_{instrument}_{channel}.fits")
        
        if isinstance(url, bytes):
            url = url.decode('utf-8')
            
        path = download_file(url)
        os.system(f'mv {path} {download_path}')
        return download_path

    except Exception as e:
        print(f"  Error downloading Spitzer data for {instrument} Ch{channel}: {e}")
        return None


def download_swift_image(coord, band_info, download_dir):
    """Queries HEASARC for Swift/UVOT data and downloads the image."""
    instrument = band_info[1]
    filter_name = band_info[0]
    print(f"  Querying HEASARC for Swift/{instrument} {filter_name}...")
    try:
        heasarc = Heasarc()
        obs_table = heasarc.query_region(coord, mission='SWIFTMASTER', radius=5 * u.arcmin)

        if not obs_table:
            print(f"  No Swift observations found in the region.")
            return None

        uvot_mask = (obs_table['INSTRUMENT'] == 'UVOT') & \
                    (obs_table['FILTER'] == filter_name) & \
                    (obs_table['IMAGETYPE'] == 'SKY')
        filtered_table = obs_table[uvot_mask]

        if len(filtered_table) == 0:
            print(f"  No suitable Swift/{instrument} {filter_name} observations found.")
            return None

        obs_id = filtered_table[0]['OBSID']
        print(f"  Found Swift OBSID: {obs_id}. Downloading data products...")

        # Download the Level 2 products for this observation
        # The heasarc.download_data method can fetch the required files.
        # We look for the sky image file, which typically ends in 'sk.img.gz'.
        downloaded_files = heasarc.download_data(obs_id, mission='SWIFT',
                                                 product_type='UVOT_IMAGE',
                                                 download_dir=download_dir)
        
        if not downloaded_files:
            print(f"  Failed to download any files for OBSID {obs_id}.")
            return None

        # Unpack the downloaded files if they are in a tar archive
        for downloaded_file in downloaded_files:
            if downloaded_file.endswith('.tar.gz'):
                with tarfile.open(downloaded_file, "r:gz") as tar:
                    tar.extractall(path=download_dir)
        
        # Search for the correct sky image file
        for root, _, files in os.walk(download_dir):
            for file in files:
                if filter_name.lower() in file.lower() and 'sk.img' in file.lower():
                    print(f"  Found and using Swift image: {file}")
                    return os.path.join(root, file)

        print(f"  No Level 2 sky image found for filter {filter_name} in OBSID {obs_id} products.")
        return None

    except Exception as e:
        print(f"  Error downloading Swift data for {instrument} {filter_name}: {e}")
        return None


# def download_xmm_image(coord, band_info, download_dir):
#     """Queries the XMM-Newton Science Archive, downloads and extracts OM image."""
#     instrument = band_info[1]
#     filter_name = band_info[0]
#     print(f"  Querying XSA for XMM-Newton/{instrument} {filter_name}...")
#     try:
#         xmm = XMMNewton()
#         obs_table = xmm.query_region(coord, radius=5 * u.arcmin)

#         if not obs_table:
#             print(f"  No XMM-Newton observations found in the region.")
#             return None
        
#         # Find an observation that actually used the OM instrument in the right filter
#         obs_id = None
#         for row in obs_table:
#             if 'OM' in row['INSTRUMENTS'] and filter_name in row['OM_FILTER']:
#                 obs_id = row['OBSERVATION_ID']
#                 break
        
#         if not obs_id:
#             print(f"  No observation found with OM using filter {filter_name}.")
#             return None

#         print(f"  Found XMM OBSID: {obs_id}. Downloading PPS data...")
        
#         # Download the PPS data pack for the OM instrument
#         xmm.download_data(observation_id=obs_id, level='PPS', inst='OM', download_dir=download_dir)

#         # Find the downloaded tar file
#         tar_path = None
#         for item in os.listdir(download_dir):
#             if item.endswith(".tar.gz") or item.endswith(".TAR"):
#                 tar_path = os.path.join(download_dir, item)
#                 break
        
#         if not tar_path:
#             print("  Could not find downloaded TAR archive.")
#             return None

#         # Extract the tar file
#         print(f"  Extracting {os.path.basename(tar_path)}...")
#         with tarfile.open(tar_path, "r:*") as tar:
#             tar.extractall(path=download_dir)
#         os.remove(tar_path) # Clean up the archive

#         # Search for the correct image file in the extracted contents
#         for root, _, files in os.walk(download_dir):
#             for file in files:
#                 # OM image files often follow this pattern
#                 if "IMAGE" in file and filter_name in file and file.endswith((".FTZ", ".fit", ".fits")):
#                     print(f"  Found image file: {file}")
#                     return os.path.join(root, file)

#         print(f"  Could not find a suitable image file for filter {filter_name} in the archive.")
#         return None

#     except Exception as e:
#         print(f"  Error downloading XMM-Newton data for {instrument} {filter_name}: {e}")
#         return None


# def query_des_catalogue(coord):
#     """Queries the DES DR1 catalogue from VizieR for photometry."""
#     print("  Querying VizieR for DES DR1 catalogue data...")
#     try:
#         v = VizieR(catalog='II/357/des_dr1', columns=['*'])
#         v.ROW_LIMIT = 1
#         result = v.query_region(coord, radius=2 * u.arcsec)

#         des_fluxes = {
#             'flux_DES_g': -999.0, 'flux_DES_r': -999.0,
#             'flux_DES_i': -999.0, 'flux_DES_z': -999.0,
#             'flux_DES_Y': -999.0
#         }

#         if not result or len(result[0]) == 0:
#             print("  No DES DR1 source found within 2 arcsec.")
#             return des_fluxes

#         source = result[0][0]
#         bands = {'g': 'gmag', 'r': 'rmag', 'i': 'imag', 'z': 'zmag', 'Y': 'Ymag'}
#         for band_key, mag_col in bands.items():
#             band_name = f'flux_DES_{band_key}'
#             if mag_col in source.columns and source[mag_col] is not np.ma.masked:
#                 magnitude = source[mag_col]
#                 flux_ujy = 3631e6 * (10**(-0.4 * magnitude))
#                 des_fluxes[band_name] = flux_ujy
#                 print(f"  Found DES {band_key}-band flux: {flux_ujy:.2f} uJy")
        
#         return des_fluxes

#     except Exception as e:
#         print(f"  Error querying DES data from VizieR: {e}")
#         return {
#             'flux_DES_g': -999.0, 'flux_DES_r': -999.0,
#             'flux_DES_i': -999.0, 'flux_DES_z': -999.0,
#             'flux_DES_Y': -999.0
#         }


def perform_photometry(image_path, coord, instrument):
    """Performs aperture photometry on a single FITS image."""
    print(f"  Performing photometry on {os.path.basename(image_path)}...")
    try:
        with fits.open(image_path, memmap=False) as hdul:
            sci_ext = 0
            if instrument in ['UVOT', 'OM']:
                sci_ext = 1
            else:
                for i, hdu in enumerate(hdul):
                    if hdu.header.get('EXTNAME') == 'SCI':
                        sci_ext = i
                        break
            
            wcs = WCS(hdul[sci_ext].header)
            data = hdul[sci_ext].data
            px, py = wcs.world_to_pixel(coord)
            position = (px, py)

            pixel_scale_deg_per_pix = proj_plane_pixel_scales(wcs)[0]
            pixel_scale_arcsec_per_pix = pixel_scale_deg_per_pix * 3600
            aperture = CircularAperture(position, r=APERTURE_RADIUS_ARCSEC / pixel_scale_arcsec_per_pix)
            
            local_background_annulus = LocalBackground(SKY_ANNULUS_INNER_ARCSEC / pixel_scale_arcsec_per_pix,
                                                    SKY_ANNULUS_OUTER_ARCSEC / pixel_scale_arcsec_per_pix)
            local_background_estimate = local_background_annulus(data, *position, mask=np.isnan(data))
            
            bkg_phot = aperture_photometry(data - local_background_estimate, aperture, wcs=wcs)

            flux_ujy = -999.0
            if 'PHOTFLAM' in hdul[sci_ext].header: # HST ABmag systems
                photflam = hdul[sci_ext].header['PHOTFLAM'] * u.erg / u.s / u.cm**2 / u.AA
                photplam = hdul[sci_ext].header['PHOTPLAM'] * u.AA
                flux_density = (bkg_phot['aperture_sum'][0] * photflam).to(u.uJy, u.spectral_density(photplam))
                flux_ujy = flux_density.value
            elif 'FLUXCONV' in hdul[sci_ext].header: # Spitzer (MJy/sr)
                flux_per_pixel_mjy = hdul[sci_ext].header['FLUXCONV']
                pixel_area_in_sq_deg = proj_plane_pixel_area(wcs) * u.deg**2
                pixel_area_sr = pixel_area_in_sq_deg.to(u.sr).value
                flux_density_mjy = bkg_phot['aperture_sum'][0] * flux_per_pixel_mjy * pixel_area_sr
                flux_ujy = (flux_density_mjy * u.MJy).to(u.uJy).value
            elif instrument in ['UVOT', 'OM']:
                if 'MAGZERO' in hdul[sci_ext].header:
                    magzero = hdul[sci_ext].header['MAGZERO']
                    if 'BUNIT' in hdul[sci_ext].header and 'COUNT' in hdul[sci_ext].header['BUNIT'].upper():
                        counts_per_sec = bkg_phot['aperture_sum'][0]
                        if counts_per_sec > 0:
                            magnitude = -2.5 * np.log10(counts_per_sec) + magzero
                            flux_ujy = 3631e6 * (10**(-0.4 * magnitude))
                        else:
                            flux_ujy = 0.0
                else:
                    print(f"  Warning: MAGZERO keyword not found for {instrument}. Cannot calculate flux.")

            print(f"  Flux: {flux_ujy:.2f} uJy")
            return flux_ujy

    except Exception as e:
        print(f"  Could not perform photometry: {e}")
        return -999.0

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

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    df_in = pd.read_csv(INPUT_CSV)
    results = []

    print(f"Starting photometry for {len(df_in)} galaxies...")

    for index, row in df_in.iterrows():
        galaxy_id = row['object_id']
        coord = SkyCoord(row['ra'], row['dec'], unit=(u.deg, u.deg))
        print(f"\nProcessing Galaxy: {galaxy_id} ({coord.to_string('hmsdms')})")

        galaxy_photometry = {'object_id': galaxy_id, 'ra': row['ra'], 'dec': row['dec']}
        
        for band_name, band_info in BANDS.items():
            query_name, instrument, archive = band_info
            
            # Create a unique, clean directory for each download attempt
            galaxy_data_dir = os.path.join(DATA_DIR, str(galaxy_id), band_name)
            os.makedirs(galaxy_data_dir, exist_ok=True)
            
            image_path = None
            if archive == 'MAST':
                image_path = download_hst_image(coord, (query_name, instrument), galaxy_data_dir)
            elif archive == 'Spitzer':
                image_path = download_spitzer_image(coord, (query_name, instrument), galaxy_data_dir)
            elif archive == 'Swift':
                image_path = download_swift_image(coord, (query_name, instrument), galaxy_data_dir)
            elif archive == 'XMM':
                image_path = download_xmm_image(coord, (query_name, instrument), galaxy_data_dir)

            if image_path and os.path.exists(image_path):
                flux = perform_photometry(image_path, coord, instrument)
                galaxy_photometry[f'flux_{band_name}'] = flux
            else:
                galaxy_photometry[f'flux_{band_name}'] = -999.0
            
            # Clean up the specific directory for this band
            cleanup_dir_contents(galaxy_data_dir)
            print(f"  Cleaned up image data directory for {band_name}.")

        des_photometry = query_des_catalogue(coord)
        galaxy_photometry.update(des_photometry)
        
        results.append(galaxy_photometry)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPhotometry complete. Results saved to {OUTPUT_CSV}")
    
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"Cleaned up temporary data directory: {DATA_DIR}")

if __name__ == "__main__":
    main()