import os
import shutil
import pandas as pd
import numpy as np
import warnings

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
from astropy.utils.data import download_file
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import LocalBackground
from regions import PolygonSkyRegion, CircleSkyRegion
from regions.core import PixCoord

# --- Configuration ---
INPUT_CSV = 'peas.csv'
OUTPUT_CSV = 'photometry_results.csv'
DATA_DIR = 'phot-temp'
APERTURE_RADIUS_ARCSEC = 2.0
SKY_ANNULUS_INNER_ARCSEC = 4.0
SKY_ANNULUS_OUTER_ARCSEC = 6.0

# Band definitions: (query_name, instrument, archive)
BANDS = {
    # HST - Using common filters for ACS and WFC3
    'HST_F218W': ('F218W', 'UVIS', 'MAST'),
    'HST_F225W': ('F225W', 'UVIS', 'MAST'),
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
        # global obs_table
        obs_table = Observations.query_criteria(
            obs_collection="HST",
            instrument_name=f"WFC3/{band_info[1]}" if "UVIS" in band_info[1] or "IR" in band_info[1] else f"{band_info[1]}/WFC",
            filters=band_info[0],
            coordinates=coord,
            t_exptime=[100, 99999], # Avoid very short exposures
            dataproduct_type="image",
            # s_region = f'CIRCLE ICRS {coord.ra.deg} {coord.dec.deg} {(10 * u.arcmin).to(u.deg).value}'
        )
        # global testcoord 
        # testcoord = coord
        
        if not obs_table:
            print(f"  No suitable HST observations found for {band_info[0]}.")
            return None
        
        if 'IR' in band_info[1]:
            polymask = np.array(['POLYGON' in item[7:] for item in obs_table['s_region'] ])
            global obs_table_clean
            obs_table_clean = obs_table[polymask]
        else:
            obs_table_clean = obs_table

        # Prioritize DRZ/DRC files (drizzled, corrected)
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
    """Queries IRSA using SIA and downloads the first available Spitzer image."""
    seip_service2= vo.dal.sia2.SIA2Service('https://irsa.ipac.caltech.edu/SIA')
    instrument = band_info[1]
    channel = band_info[0]
    print(f"  Querying IRSA for Spitzer/{instrument} Ch{channel}...")
    try:
        # Use the Simple Image Access (SIA) protocol to find images
        # The SIA service requires the radius to be in the 'pos' tuple
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

        # Get the download URL for the mosaic FITS file
        url = filtered_table[0]['access_url']
        download_path = os.path.join(download_dir, f"spitzer_{instrument}_{channel}.fits")
        
        # astroquery returns URLs as bytes, so decode them
        if isinstance(url, bytes):
            url = url.decode('utf-8')
            
        path = download_file(url)
        os.system(f'mv {path} {download_path}')
        return download_path

    except Exception as e:
        print(f"  Error downloading Spitzer data for {instrument} Ch{channel}: {e}")
        return None

def perform_photometry(image_path, coord, instrument):
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
            
            # print(f'sci extension {sci_ext}')

            wcs = WCS(hdul[sci_ext].header)
            data = hdul[sci_ext].data

            # Convert RA/Dec to pixel coordinates
            px, py = wcs.world_to_pixel(coord)
            position = (px, py)

            # Define apertures
            pixel_scale_deg_per_pix = proj_plane_pixel_scales(wcs)[0]
            pixel_scale_arcsec_per_pix = pixel_scale_deg_per_pix * 3600  # Convert deg to arcsec
            aperture = CircularAperture(position, r=APERTURE_RADIUS_ARCSEC / pixel_scale_arcsec_per_pix)
            # annulus = CircularAnnulus(position,
                                        # r_in=SKY_ANNULUS_INNER_ARCSEC / pixel_scale_arcsec_per_pix,
                                        # r_out=SKY_ANNULUS_OUTER_ARCSEC / pixel_scale_arcsec_per_pix)
            local_background_annulus = LocalBackground(SKY_ANNULUS_INNER_ARCSEC / pixel_scale_arcsec_per_pix,
                                                    SKY_ANNULUS_OUTER_ARCSEC / pixel_scale_arcsec_per_pix)
            local_background_estimate = local_background_annulus(data,*position, mask=np.isnan(data))
            # Perform background-subtracted photometry
            # bkg_phot = aperture_photometry(data - local_background_estimate, aperture, wcs=wcs)
            bkg_phot = aperture_photometry(data, aperture, wcs=wcs)

            # print(local_background_estimate)
            # print(bkg_phot)
            # Unit conversion to microjanskys (uJy)
            flux_ujy = -999.0
            if 'PHOTFLAM' in hdul[sci_ext].header: # HST ABmag systems
                # print('test')
                photflam = hdul[sci_ext].header['PHOTFLAM'] * u.erg / u.s / u.cm**2 / u.AA
                photplam = hdul[sci_ext].header['PHOTPLAM'] * u.AA
                # print(photflam,photplam)
                flux_density = (bkg_phot['aperture_sum'][0] * photflam).to(u.uJy, u.spectral_density(photplam))
                # print(flux_density)
                flux_ujy = flux_density.value
            elif 'FLUXCONV' in hdul[sci_ext].header: # Spitzer (MJy/sr)
                flux_per_pixel_mjy = hdul[sci_ext].header['FLUXCONV']
                pixel_area_in_sq_deg = proj_plane_pixel_area(wcs) * u.deg**2
                pixel_area_sr = pixel_area_in_sq_deg.to(u.sr).value
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
        galaxy_id = row['object_id']
        coord = SkyCoord(row['ra'], row['dec'], unit=(u.deg, u.deg))
        print(f"\nProcessing Galaxy: {galaxy_id} ({coord.to_string('hmsdms')})")

        galaxy_photometry = {'object_id': galaxy_id, 'ra': row['ra'], 'dec': row['dec']}
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
                flux = perform_photometry(image_path, coord, instrument)
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


# def parse_s_region(s_region_string: str) -> list[PolygonSkyRegion]:
#     """
#     Parses an s_region string from an astronomical catalog query into a list
#     of astropy PolygonSkyRegion objects.

#     Handles strings containing one or multiple POLYGON definitions.
#     """
#     # The string can contain multiple polygons, so we split by the keyword "POLYGON"
#     # This will result in a list where the first element is empty.
#     polygon_strs = s_region_string.strip().upper().split('POLYGON')[1:]
    
#     regions = []
#     for poly_str in polygon_strs:
#         # The first word might be a coordinate system (e.g., ICRS).
#         # We need to determine if the first element is a string or a coordinate.
#         parts = poly_str.strip().split()
#         if not parts:
#             continue

#         # Check if the first part is the coordinate system or the first coordinate
#         try:
#             # If this succeeds, the first part is a coordinate, and there is no system string
#             float(parts[0])
#             coord_parts = parts
#         except (ValueError, IndexError):
#             # Otherwise, the first part is the system string (e.g., 'ICRS'), so we skip it
#             coord_parts = parts[1:]

#         # Extract coordinate values (as floats)
#         coords_flat = [float(p) for p in coord_parts]

#         # A valid polygon must have an even number of coordinates. If not, skip it.
#         if len(coords_flat) % 2 != 0:
#             # Consider logging a warning here in a real application
#             continue

#         # Group the flat list into pairs of (ra, dec)
#         vertices_coords = np.reshape(coords_flat, (-1, 2))
        
#         # Create an astropy SkyCoord object for the vertices
#         vertices = SkyCoord(vertices_coords, unit='deg', frame='icrs')
        
#         # Create the PolygonSkyRegion and add it to our list
#         regions.append(PolygonSkyRegion(vertices=vertices))
        
#     return regions

# def is_circle_in_footprint(
#     s_region_string: str,
#     circle_center: SkyCoord,
#     circle_radius: u.Quantity
# ) -> bool:
#     """
#     Checks if a circular aperture is fully contained within an observation footprint.

#     The footprint can consist of one or more polygons (e.g., for multi-chip detectors).
#     The circle is considered "in" if it is fully contained by ANY of the polygons.

#     Args:
#         s_region_string: The string value from the 's_region' column.
#         circle_center: The center of the circular aperture as an astropy SkyCoord.
#         circle_radius: The radius of the aperture as an astropy Quantity (e.g., 1 * u.arcsec).

#     Returns:
#         True if the circle is contained in any of the footprint's polygons, False otherwise.
#     """
#     if not s_region_string or not isinstance(s_region_string, str):
#         return False

#     # Create the circular region for your object of interest
#     circle_to_check = CircleSkyRegion(center=circle_center, radius=circle_radius)
    
#     # Parse the footprint string into one or more PolygonSkyRegion objects
#     footprint_polygons = parse_s_region(s_region_string)

#     if not footprint_polygons:
#         return False

#     # To check if a SkyRegion contains another, we need a WCS object to project
#     # the regions onto a common 2D plane. We can create a simple tangential
#     # projection centered on the region of interest.
#     wcs = WCS(naxis=2)
#     wcs.wcs.crpix = [0, 0]
#     # Use a pixel scale appropriate for the circle size, e.g., 1/10th of the radius
#     pixel_scale = (circle_radius / 10).to(u.deg).value
#     wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale])
#     # Center the projection on the circle's center for accuracy
#     wcs.wcs.crval = [circle_center.ra.deg, circle_center.dec.deg]
#     wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
#     # Check if the circle is contained in *any* of the polygons
#     def contains(self, skycoord, wcs):
#         pixel_region = self.to_pixel(wcs)
#         # pixcoord = PixCoord.from_sky(skycoord, wcs)
#         x,y = skycoord.to_pixel(wcs=wcs)
#         pixcoord = PixCoord(x=x,y=y)
#         return pixel_region.contains(pixcoord)
    
#     for polygon in footprint_polygons:
#         # if polygon.contains(circle_to_check, wcs=wcs):
#         if contains(polygon,circle_to_check,wcs=wcs):
#             return True
            
#     return False

# def check_footprints_in_table(
#     table: Table,
#     circle_center: SkyCoord,
#     circle_radius: u.Quantity,
#     s_region_col: str = 's_region'
# ) -> np.ndarray:
#     """
#     Efficiently checks which rows in an Astropy Table contain a circular aperture.

#     This function iterates over the table's s_region column and returns a
#     boolean numpy array that can be used to mask the table.

#     Args:
#         table: The Astropy Table containing the observation footprints.
#         circle_center: The center of the circular aperture (SkyCoord).
#         circle_radius: The radius of the aperture (astropy Quantity).
#         s_region_col: The name of the column containing the s_region strings.

#     Returns:
#         A numpy array of booleans with the same length as the table.
#         True where the circle is contained, False otherwise.
#     """
#     is_contained_list = [
#         is_circle_in_footprint(row[s_region_col], circle_center, circle_radius)
#         for row in table
#     ]
#     return np.array(is_contained_list)


if __name__ == "__main__":
    main()

