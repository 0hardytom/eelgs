import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm

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


def plot_hubble_with_spitzer_contours(hubble_file, spitzer_file):
    """
    Plots a Hubble image with Spitzer image contours overlaid.

    Args:
        hubble_file (str): Path to the Hubble FITS file.
        spitzer_file (str): Path to the Spitzer FITS file.
    """
    # --- Load the FITS files ---
    # Open the Hubble file to get the data and WCS for the main plot
    with fits.open(hubble_file) as hdul:
        hubble_data = hdul[0].data
        hubble_wcs = WCS(hdul[0].header)

    # Open the Spitzer file to get the data and WCS for the contours
    with fits.open(spitzer_file) as hdul:
        spitzer_data = hdul[0].data
        spitzer_wcs = WCS(hdul[0].header)

    # --- Create the Plot ---
    # Initialize the plot using the Hubble WCS projection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=hubble_wcs)

    # --- Display the Hubble Image ---
    # Use simple_norm for automatic scaling of the image display
    norm = simple_norm(hubble_data, 'sqrt', percent=99.5)
    ax.imshow(hubble_data, origin='lower', cmap='gray_r', norm=norm)

    # --- Overlay Spitzer Contours ---
    # The key is the 'transform' argument. It tells matplotlib to draw the
    # spitzer_data contours by interpreting their pixel coordinates using the
    # spitzer_wcs, and then transforming them to the Hubble WCS of the plot.
    ax.contour(
        spitzer_data,
        transform=ax.get_transform(spitzer_wcs),
        levels=np.logspace(1, 2.5, 8), # Example contour levels
        colors='cyan',
        linewidths=0.8
    )

    # --- Final Touches ---
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_title('Spitzer Contours on Hubble Image')
    ax.grid(color='white', ls=':', alpha=0.5)

    plt.savefig('hubble_spitzer_plot.png', dpi=300)
    print("\nSaved plot to hubble_spitzer_plot.png")
    plt.show()


if __name__ == '__main__':
    # 1. Generate dummy files for the example to work out-of-the-box
    create_dummy_fits_files()

    # 2. Define the filenames.
    #    !!! IMPORTANT: Replace these with your actual file paths !!!
    hubble_filename = 'dummy_hubble.fits'
    spitzer_filename = 'dummy_spitzer.fits'

    # 3. Run the plotting function
    plot_hubble_with_spitzer_contours(hubble_filename, spitzer_filename)
