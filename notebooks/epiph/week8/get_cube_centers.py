

import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import numpy as np

# Get the list of FITS files in the current directory
files = [f for f in os.listdir('.') if f.endswith('.fits') and not f.startswith('._')]

# Initialize lists to store the data
filenames = []
ras = []
decs = []

# Loop over the files
for filename in files:
    # Open the FITS file
    with fits.open(filename) as hdul:
        # Get the header from the primary HDU
        header = hdul[1].header
        
        # Create a WCS object from the header
        w = WCS(header)
        
        # Get the image size
        nx = header['NAXIS1']
        ny = header['NAXIS2']
        
        # Calculate the center pixel
        center_pix = np.array([[nx/2., ny/2.,0]])
        
        # Convert pixel coordinates to world coordinates
        center_world = w.wcs_pix2world(center_pix,1)
        
        # Append the data to the lists
        filenames.append(filename.replace('_COMBINED_CUBE_MED_FINAL.fits',''))
        ras.append(center_world[0][0])
        decs.append(center_world[0][1])
        
        print(f"Processed {filename}")


# Create an astropy Table
t = Table([filenames, ras, decs], names=('filename', 'ra', 'dec'))

# Write the table to a file
t.write('cube_centers.csv', overwrite=True)

print("\nFinished processing all files.")
print("The results have been written to cube_centers.txt")

