
import os
import glob
import csv
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

def get_central_coord(header):
    """
    Calculate the central RA and Dec from the FITS header WCS.
    """
    try:
        wcs = WCS(header)
        naxis = header.get('NAXIS', 2)
        nx = header['NAXIS1']
        ny = header['NAXIS2']

        if naxis == 2:
            # 2D Image
            center_pix = np.array([[nx/2, ny/2]])
        elif naxis == 3:
            # 3D Cube, use the center of the third axis
            nz = header['NAXIS3']
            center_pix = np.array([[nx/2, ny/2, nz/2]])
        else:
            print(f"Unsupported number of axes ({naxis}) for {header.get('OBJECT', 'Unknown')}")
            return None, None

        # Transform pixel to world coordinates
        center_world = wcs.all_pix2world(center_pix, 1)
        
        # The result may have 3 dimensions (RA, Dec, Wavelength), we only want the first two.
        return center_world[0][0], center_world[0][1]
    except Exception as e:
        print(f"Could not determine WCS for {header.get('OBJECT', 'Unknown')}: {e}")
        return None, None

def find_and_process_cubes(base_dirs, output_csv):
    """
    Finds all FITS cubes, extracts header info, and writes to a CSV.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'object', 'central_ra', 'central_dec', 'source_dir']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_files = 0
        for base_dir in base_dirs:
            pattern = os.path.join(base_dir, '**', '*_COMBINED_CUBE_MED_FINAL.fits')
            cube_files = glob.glob(pattern, recursive=True)
            total_files += len(cube_files)
            source_dir_name = os.path.basename(base_dir)

            for fpath in cube_files:
                try:
                    with fits.open(fpath) as hdul:
                        # Try the first extension first, as primary is often empty
                        if len(hdul) > 1:
                            header = hdul[1].header
                        else:
                            header = hdul[0].header
                        
                        # Split at '(' and take the first part to remove any trailing info
                        object_name = header.get('OBJECT', 'Unknown').split('(')[0].strip()
                        ra, dec = get_central_coord(header)

                        if ra is not None and dec is not None:
                            # Extract just the ID from the start of the filename
                            file_id = os.path.basename(fpath).split('_')[0]
                            writer.writerow({
                                'filename': file_id,
                                'object': object_name,
                                'central_ra': ra,
                                'central_dec': dec,
                                'source_dir': source_dir_name
                            })
                except Exception as e:
                    print(f"Error processing file {fpath}: {e}")
        
        print(f"Found and processed {total_files} cubes.")

if __name__ == '__main__':
    base_directories = ['/Volumes/Expansion/exp_thardy/cubes', '/Volumes/Expansion/exp_thardy/cubes_new']
    output_file = 'cube_info.csv'
    find_and_process_cubes(base_directories, output_file)
    print(f"Processing complete. Output written to {output_file}")
