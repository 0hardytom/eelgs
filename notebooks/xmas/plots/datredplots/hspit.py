import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from reproject import reproject_interp
import astropy.units as u
from astropy.coordinates import SkyCoord

# (The create_dummy_fits_files function remains the same)
def create_dummy_fits_files():
    """
    Generates two dummy FITS files (Hubble and Spitzer) with basic WCS info.
    This allows the script to be run without needing real data immediately.
    """
    # --- Create a dummy Hubble FITS file ---
    # Create a basic WCS header
    wcs_hubble = WCS(naxis=2)
    wcs_hubble.wcs.crpix = [256, 256]  # Reference pixel
    wcs_hubble.wcs.cdelt = np.array([-0.05 / 3600, 0.05 / 3600]) # Pixel scale in degrees
    wcs_hubble.wcs.crval = [150.119, 2.201]  # RA/Dec of reference pixel
    wcs_hubble.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header_hubble = wcs_hubble.to_header()

    # Create dummy data (e.g., a compact source)
    y, x = np.mgrid[:512, :512]
    data_hubble = np.exp(-((x - 256)**2 + (y - 256)**2) / (2 * 15**2)) * 100
    data_hubble += np.random.normal(loc=0, scale=2, size=(512, 512)) # Add some noise

    # Write to FITS file
    hdu_hubble = fits.PrimaryHDU(data=data_hubble, header=header_hubble)
    hdu_hubble.writeto('dummy_hubble.fits', overwrite=True)
    print("Created dummy_hubble.fits")

    # --- Create a dummy Spitzer FITS file (lower resolution) ---
    wcs_spitzer = WCS(naxis=2)
    wcs_spitzer.wcs.crpix = [64, 64]
    wcs_spitzer.wcs.cdelt = np.array([-0.2 / 3600, 0.2 / 3600]) # Larger pixel scale
    wcs_spitzer.wcs.crval = [150.119, 2.201] # Same center for alignment
    wcs_spitzer.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header_spitzer = wcs_spitzer.to_header()

    # Create dummy data (e.g., a more diffuse source)
    y, x = np.mgrid[:128, :128]
    data_spitzer = np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 10**2)) * 50
    data_spitzer += np.random.normal(loc=0, scale=1, size=(128, 128))

    # Write to FITS file
    hdu_spitzer = fits.PrimaryHDU(data=data_spitzer, header=header_spitzer)
    hdu_spitzer.writeto('dummy_spitzer.fits', overwrite=True)
    print("Created dummy_spitzer.fits")


def plot_zoomed(hubble_file, spitzer_file, center_ra, center_dec, radius_arcmin):
    """
    Plots a zoomed-in view of the Hubble image with Spitzer contours.

    Args:
        hubble_file (str): Path to the Hubble FITS file.
        spitzer_file (str): Path to the Spitzer FITS file.
        center_ra (float): Central Right Ascension in degrees.
        center_dec (float): Central Declination in degrees.
        radius_arcmin (float): Radius of the desired plot view in arcminutes.
    """
    # --- Load and reproject data (same as before) ---
    with fits.open(hubble_file) as hdul:
        hubble_data = hdul[0].data
        hubble_wcs = WCS(hdul[0].header)
    with fits.open(spitzer_file) as hdul:
        spitzer_data = hdul[0].data
        spitzer_header = hdul[0].header

    if hubble_data.ndim == 3: hubble_data = hubble_data[0, :, :]
    if spitzer_data.ndim == 3: spitzer_data = spitzer_data[0, :, :]

    spitzer_reprojected, _ = reproject_interp(
        (spitzer_data, spitzer_header),
        hubble_wcs,
        shape_out=hubble_data.shape
    )

    # --- Create the Plot ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=hubble_wcs)

    # --- Display the Hubble Image ---
    norm = simple_norm(hubble_data, 'sqrt', percent=99.5)
    ax.imshow(hubble_data, origin='lower', cmap='gray_r', norm=norm)

    # --- Overlay Spitzer Contours ---
    ax.contour(
        spitzer_reprojected,
        levels=np.logspace(-1, 2, 8),
        colors='cyan',
        linewidths=0.8
    )

    # --- NEW: SET THE PLOT VIEW ---
    # 1. Define the center coordinate and radius
    center_coord = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg, frame='icrs')
    plot_radius = radius_arcmin * u.arcmin

    # 2. Calculate the RA and Dec limits for the plot
    #    The plot will be a square with width and height of 2 * plot_radius
    ra_lim = (center_coord.ra - plot_radius, center_coord.ra + plot_radius)
    dec_lim = (center_coord.dec - plot_radius, center_coord.dec + plot_radius)

    # 3. Apply the limits to the axes
    #    For WCSAxes, set_xlim and set_ylim expect world coordinates
    ax.set_xlim(ra_lim)
    ax.set_ylim(dec_lim)
    # --- END OF NEW CODE ---

    # --- Final Touches ---
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_title(f'Zoomed View (Radius: {radius_arcmin}\')')
    ax.grid(color='white', ls=':', alpha=0.5)

    plt.savefig('hubble_spitzer_zoomed_plot.png', dpi=300)
    print(f"\nSaved plot to hubble_spitzer_zoomed_plot.png")
    plt.show()


if __name__ == '__main__':
    # 1. Generate dummy files for the example
    create_dummy_fits_files()

    # 2. Define filenames
    hubble_filename = 'dummy_hubble.fits'
    spitzer_filename = 'dummy_spitzer.fits'

    # 3. !!! SET YOUR DESIRED VIEW HERE !!!
    #    Using the center of the dummy image as an example
    center_ra_deg = 150.119
    center_dec_deg = 2.201
    radius_arcmin = 0.5 # Set your desired radius in arcminutes

    # 4. Run the plotting function
    plot_zoomed(
        hubble_filename,
        spitzer_filename,
        center_ra_deg,
        center_dec_deg,
        radius_arcmin
    )
