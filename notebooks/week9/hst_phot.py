
import argparse
import os

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry


def query_hst_imaging(ra, dec, radius=0.1):
    """
    Queries the MAST archive for HST imaging data.
    """
    coords = SkyCoord(ra, dec, unit="deg")
    obs_table = Observations.query_region(coords, radius=radius)
    # Filter for HST imaging data
    hst_imaging = obs_table[(obs_table['obs_collection'] == 'HST') & (obs_table['dataproduct_type'] == 'image')]
    return hst_imaging

def download_hst_images(obs_table, data_dir='hst_data'):
    """
    Downloads HST images from the MAST archive.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    manifest = Observations.download_products(obs_table['obs_id'], download_dir=data_dir)
    return [os.path.join(data_dir, row['Local Path']) for row in manifest]


def perform_photometry(image_path, ra, dec):
    """
    Performs aperture photometry on a FITS image.
    """
    with fits.open(image_path) as hdul:
        wcs = WCS(hdul[1].header)
        data = hdul[1].data
        
        coords = SkyCoord(ra, dec, unit="deg")
        pixel_coords = wcs.world_to_pixel(coords)
        
        positions = [(pixel_coords[0], pixel_coords[1])]
        aperture = CircularAperture(positions, r=6.0)
        
        phot_table = aperture_photometry(data, aperture)
        return phot_table['aperture_sum'][0]


def main():
    parser = argparse.ArgumentParser(description='Query HST imaging and perform photometry for a catalog of galaxies.')
    parser.add_argument('table_file', type=str, help='Path to the astropy table file.')
    args = parser.parse_args()

    data = Table.read(args.table_file, format='ascii')

    if 'ra' not in data.colnames or 'dec' not in data.colnames:
        raise ValueError("Input table must contain 'ra' and 'dec' columns.")

    for row in data:
        ra = row['ra']
        dec = row['dec']
        
        print(f"Processing object at (RA={ra}, Dec={dec})...")
        
        obs_table = query_hst_imaging(ra, dec)
        if not obs_table:
            print(f"No HST imaging data found for object at (RA={ra}, Dec={dec}).")
            continue
        
        image_paths = download_hst_images(obs_table)
        
        for image_path in image_paths:
            try:
                flux = perform_photometry(image_path, ra, dec)
                print(f"  - Estimated flux in {os.path.basename(image_path)}: {flux}")
            except Exception as e:
                print(f"  - Error processing {os.path.basename(image_path)}: {e}")

