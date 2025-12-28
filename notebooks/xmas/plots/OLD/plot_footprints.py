import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.patches import Polygon

def plot_muse_footprints_final():
    """
    Finds all MUSE FITS cubes and plots their footprints and center points
    on a clean, reliable all-sky Mollweide projection. This version removes
    complex backgrounds and custom styling to ensure stability.
    """
    # --- 1. Find all FITS files and extract their footprints ---
    search_path = os.path.join('cubes', '**', '*.fits*')
    cube_files = glob.glob(search_path, recursive=True)

    if not cube_files:
        print("Error: No FITS files found in the 'cubes/' directory.")
        return

    print(f"Found {len(cube_files)} FITS files to process.")
    all_footprints_deg = []
    all_centers_deg = []

    for i, f in enumerate(cube_files):
        try:
            with fits.open(f) as hdul:
                wcs_header = None
                for hdu in hdul:
                    if (hdu.header.get('CTYPE1') and hdu.header.get('CTYPE2') and
                        hdu.header.get('NAXIS', 0) >= 2):
                        wcs_header = hdu.header
                        break
                if wcs_header is None:
                    raise ValueError("No valid WCS found.")

                wcs = WCS(wcs_header)
                footprint = wcs.celestial.calc_footprint()
                center = wcs.celestial.wcs.crval
                all_footprints_deg.append(footprint)
                all_centers_deg.append(center)
        except Exception as e:
            print(f"Warning: Could not process file {os.path.basename(f)}. Reason: {e}")
            continue

    if not all_centers_deg:
        print("Error: Could not extract valid WCS information from any FITS files.")
        return

    # --- 2. Create the all-sky plot ---
    print("\nGenerating all-sky plot...")
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="mollweide")

    # --- Plot markers and footprints ---
    # Plot a visible marker for the center of each field
    for center_deg in all_centers_deg:
        # Convert RA to radians in the range [-pi, pi] for Mollweide
        ra_rad = np.deg2rad(center_deg[0])
        ra_rad = ra_rad if ra_rad <= np.pi else ra_rad - 2 * np.pi
        dec_rad = np.deg2rad(center_deg[1])
        ax.plot(ra_rad, dec_rad, 'o', color='red', markersize=5, alpha=0.8)

    # Plot the (likely very small) footprints
    for footprint_deg in all_footprints_deg:
        ra_rad = np.deg2rad(footprint_deg[:, 0])
        ra_rad[ra_rad > np.pi] -= 2 * np.pi
        dec_rad = np.deg2rad(footprint_deg[:, 1])
        footprint_rad = np.column_stack((ra_rad, dec_rad))
        poly = Polygon(footprint_rad, edgecolor='cyan', facecolor='cyan',
                       alpha=0.6, linewidth=1)
        ax.add_patch(poly)

    # --- 3. Final plot formatting ---
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_title('All-Sky Distribution of MUSE Fields')
    ax.grid(True, color='gray', linestyle='--', alpha=0.7)

    # Set RA tick labels to hours
    ax.set_xticklabels(['10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h'])
    
    # Set Dec tick labels to degrees
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))

    # --- 4. Save the plot ---
    output_filename = 'muse_fields_allsky_overview.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {output_filename}")

if __name__ == '__main__':
    plot_muse_footprints_final()