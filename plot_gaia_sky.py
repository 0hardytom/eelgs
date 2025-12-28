
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astroquery.hips2fits import hips2fits
import numpy as np
import warnings
from matplotlib.colors import Normalize
from reproject import reproject_interp
from astropy.io import fits
import astropy.units as u

# Suppress warnings from astropy WCS
warnings.filterwarnings('ignore', category=UserWarning, append=True)


def plot_gaia_sky():
    """
    Fetches a simple Cartesian projection of the Gaia EDR3 density map and
    reprojects it onto a locally-created, correct Mollweide projection in
    Galactic coordinates, ensuring axis labels are drawn and visible.
    """
    print("Querying HiPS2FITS service for raw Cartesian data...")
    try:
        # Fetch a simple Cartesian projection (Plate Carrée)
        input_hdu = hips2fits.query(
            hips='CDS/P/DM/I/350/gaiaedr3',
            width=4096,
            height=2048,
            ra=0 * u.deg,
            dec=0 * u.deg,
            fov=360 * u.deg,
            projection='MOL',
            coordsys='galactic',
            format='fits'
        )[0]
    except Exception as e:
        print(f"Error querying HiPS2FITS service: {e}")
        return

    print("Creating a new, correct Mollweide projection header...")
    target_header = fits.Header()
    target_header['NAXIS'] = 2
    target_header['NAXIS1'] = 2048
    target_header['NAXIS2'] = 1024
    target_header['CTYPE1'] = 'GLON-MOL'
    target_header['CRPIX1'] = 1024.5
    target_header['CRVAL1'] = 0.0
    target_header['CDELT1'] = -0.175
    target_header['CUNIT1'] = 'deg'
    target_header['CTYPE2'] = 'GLAT-MOL'
    target_header['CRPIX2'] = 512.5
    target_header['CRVAL2'] = 0.0
    target_header['CDELT2'] = 0.175
    target_header['CUNIT2'] = 'deg'
    
    target_wcs = WCS(target_header)

    print("Reprojecting image to the new Mollweide projection...")
    output_array, footprint = reproject_interp(input_hdu, target_wcs, shape_out=(1024, 2048))

    print("Processing final image data...")
    output_array[np.isnan(output_array)] = 0
    with np.errstate(divide='ignore'):
        plot_data = np.log1p(output_array)

    print("Creating final plot...")
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection=target_wcs)

    vmin = np.percentile(plot_data[plot_data > 0], 1)
    vmax = np.percentile(plot_data, 99.5)
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(plot_data, origin='lower', cmap='hot', norm=norm)

    cbar = fig.colorbar(im, orientation='horizontal', pad=0.1, aspect=40)
    cbar.set_label('Log(1 + Source Density)')

    ax.coords.grid(True, color='white', ls='solid', alpha=0.3)
    lon, lat = ax.coords['glon'], ax.coords['glat']
    
    lon.set_axislabel('Galactic Longitude')
    lat.set_axislabel('Galactic Latitude')

    lon.set_major_formatter('d')
    lat.set_major_formatter('d')

    # Explicitly configure the ticks and labels to be white so they are visible
    lon.set_ticks(spacing=30 * u.degree, color='white')
    lat.set_ticks(spacing=30 * u.degree, color='white')
    lon.set_ticklabel(color='white', exclude_overlapping=True)
    lat.set_ticklabel(color='white', exclude_overlapping=True)

    ax.set_title("Gaia EDR3 All-Sky Density Map (Galactic)", pad=20)
    
    # Use tight_layout to prevent labels from being clipped
    plt.tight_layout(pad=2.5)

    output_filename = 'gaia_mollweide_galactic_reprojected.png'
    print(f"Saving plot to {output_filename}...")
    plt.savefig(output_filename, dpi=300)
    print("Plot saved successfully.")
    plt.close()

    return input_hdu 


if __name__ == '__main__':
    plot_gaia_sky()
