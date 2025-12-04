
import os
import shutil
import tempfile
import numpy as np

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry


def query_hst_imaging(ra, dec, radius=0.1, filters=['F435W', 'F606W', 'F814W',
                                                    'F105W', 'F125W', 'F160W']):
    """
    Queries the MAST archive for HST imaging data in specific filters.
    """
    coords = SkyCoord(ra, dec, unit="deg")
    obs_table = Observations.query_region(coords, radius=radius)
    
    # Filter for HST imaging data and specified filters
    hst_imaging = obs_table[
        (obs_table['obs_collection'] == 'HST') & 
        (obs_table['dataproduct_type'] == 'image') &
        np.isin(obs_table['filters'], filters)
    ]

    if len(hst_imaging) == 0:
        return hst_imaging

    # Select one observation per filter to avoid duplicates.
    unique_filters = {}
    indices_to_keep = []
    for i, row in enumerate(hst_imaging):
        if row['filters'] not in unique_filters:
            unique_filters[row['filters']] = True
            indices_to_keep.append(i)
    
    return hst_imaging[indices_to_keep]

def download_hst_images(obs_table, data_dir):
    """
    Downloads HST science images from the MAST archive into a specific directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Get product list and filter for science FITS files
    products = Observations.get_product_list(obs_table)
    
    science_mask = ((products['productType'] == 'SCIENCE') & 
    np.isin(products['productSubGroupDescription'], ['DRZ', 'DRC']) & 
    [fn.endswith('.fits') for fn in products['productFilename']])
    science_products = products[science_mask]
    
    if len(science_products) == 0:
        print("  - No science FITS images found in products.")
        return []

    manifest = Observations.download_products(science_products, download_dir=data_dir)
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


def process_table(data_table, filters_to_process=['F475W', 'F625W', 'F814W']):
    """
    Processes an astropy table of objects, performing HST photometry for each.
    Returns a copy of the table with new columns for HST filter fluxes.
    """
    if 'ra' not in data_table.colnames or 'dec' not in data_table.colnames:
        raise ValueError("Input table must contain 'ra' and 'dec' columns.")

    new_table = data_table.copy()
    all_fluxes = []
    all_filters = set()

    temp_dir = tempfile.mkdtemp()
    try:
        for i, row in enumerate(new_table):
            ra = row['ra']
            dec = row['dec']
            
            print(f"Processing object {i+1}/{len(new_table)} at (RA={ra}, Dec={dec})...")
            
            obs_table = query_hst_imaging(ra, dec, filters=filters_to_process)
            if not obs_table or len(obs_table) == 0:
                print(f"  - No HST imaging data found for specified filters.")
                all_fluxes.append({})
                continue
            
            filters = obs_table['filters']
            all_filters.update(filters)
            
            image_paths = download_hst_images(obs_table, temp_dir)
            
            row_fluxes = {}
            for i, image_path in enumerate(image_paths):
                try:
                    flux = perform_photometry(image_path, ra, dec)
                    # Find the corresponding filter from the observation table
                    # This is more robust than relying on download order
                    fname = os.path.basename(image_path)
                    product_row = obs_table[obs_table['obs_id'] == fname.split('_')[0]]
                    filter_name = product_row['filters'][0]
                    row_fluxes[filter_name] = flux
                except Exception as e:
                    print(f"  - Error processing {os.path.basename(image_path)}: {e}")
            all_fluxes.append(row_fluxes)
    finally:
        shutil.rmtree(temp_dir)

    # Add new columns for each unique filter found
    for f in sorted(list(all_filters)):
        # Sanitize filter name to be a valid column name
        col_name = f"flux_{f.replace('/', '_').replace(';', '_')}"
        new_table[col_name] = [fluxes.get(f, np.nan) for fluxes in all_fluxes]
        
    return new_table
