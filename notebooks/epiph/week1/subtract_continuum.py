
import numpy as np
from mpdaf.obj import Cube
from scipy.ndimage import median_filter
import sys
import os

def subtract_continuum(input_fits_path, output_fits_path, filter_width=151):
    """
    Performs a running median filter subtraction on a FITS datacube.

    This function reads a 3D FITS datacube, calculates the running median
    along the spectral axis, and subtracts it to remove the continuum.

    Args:
        input_fits_path (str): Path to the input FITS datacube.
        output_fits_path (str): Path to save the continuum-subtracted FITS file.
        filter_width (int): The full width of the median filter in pixels (spectral layers).
                            Must be an odd number.
    """
    if filter_width % 2 == 0:
        raise ValueError("filter_width must be an odd number.")

    print(f"Opening FITS file: {input_fits_path}")
    try:
        cube = Cube(input_fits_path)
        
        # Verify that the data is a 3D cube
        if cube.data.ndim != 3:
            raise ValueError(f"Expected a 3D datacube, but got {cube.data.ndim} dimensions.")

        print(f"Original data shape (wavelength, y, x): {cube.data.shape}")
        print(f"Applying a running median filter of width {filter_width} along the spectral axis...")

        # The filter size is (width, 1, 1) to apply it ONLY along the first (spectral) axis.
        # The 'mode' parameter handles how the edges of the data are treated.
        continuum_model = median_filter(
            cube.data,
            size=(filter_width, 1, 1),
            mode='reflect'
        )

        print("Subtracting the continuum model from the original data...")
        subtracted_cube_data = cube.data - continuum_model

        print(f"Saving the continuum-subtracted cube to: {output_fits_path}")
        # Create a new cube with the subtracted data and original header
        new_cube = cube.new_from_obj(cube)
        new_cube.data = subtracted_cube_data

        # Write the new FITS file
        new_cube.write(output_fits_path, savemask='nan')
        print("Done.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_fits_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # This allows you to run the script from the command line.
    # Example usage:
    # python subtract_continuum.py /path/to/input_dir /path/to/output_dir
    if len(sys.argv) != 3:
        print("Usage: python subtract_continuum.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"Output directory not found at {output_dir}, creating it.")
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".fits") or filename.endswith(".fit"):
            input_file = os.path.join(input_dir, filename)
            base, ext = os.path.splitext(filename)
            output_file = os.path.join(output_dir, f"{base}_CSUB{ext}")
            subtract_continuum(input_file, output_file)
