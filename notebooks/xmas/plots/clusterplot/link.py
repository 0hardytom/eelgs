
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.wcsutil import proj_plane_pixel_scales
from astropy.visualization import make_lupton_rgb
import astropy.units as u

# Assuming 'image_r', 'image_g', 'image_b' are FITS HDU objects or similar
# with .data and .wcs attributes.
# Assuming 'pf' is a custom plotting helper module.
# Assuming 'crop' is a custom function.

# Placeholder for missing objects - replace with your actual data
class MockHDU:
    def __init__(self):
        import numpy as np
        from astropy.wcs import WCS
        self.data = np.random.rand(100, 100)
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [50, 50]
        self.wcs.wcs.cdelt = np.array([-0.1, 0.1])
        self.wcs.wcs.crval = [8.478, -7.866]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

image_r = MockHDU()
image_g = MockHDU()
image_b = MockHDU()

def crop(data, n):
    return data[n:-n, n:-n]

class PlottingFunctions:
    def fix_plot(self, axes, tickdir):
        pass # Placeholder

pf = PlottingFunctions()


# Original plotting code
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1, projection=image_r.wcs.wcs)

specx = fig.add_axes((1.0,0.1,0.7,0.2))
specx2 = fig.add_axes((1.0,0.3,0.7,0.1))
contx = fig.add_axes((1.7,0.1,0.3,0.3))

axes = [ax,specx,specx2,contx]

for a in [specx2,contx]:
    plt.setp(a.get_xticklabels(), visible=False)
    plt.setp(a.get_yticklabels(), visible=False)

    a.set_xticks([])
    a.set_yticks([])

cropn = 23

rgbimage  = make_lupton_rgb(crop(image_r.data, cropn),crop(image_g.data,cropn), crop(image_b.data,cropn),
                                stretch=20, Q=1, minimum=.1,)
ax.imshow(rgbimage, origin='lower')
ax.invert_xaxis()

ra = ax.coords[0]
dec = ax.coords[1]

ra.set_axislabel('Right Ascension')
dec.set_axislabel('Declination')


galloc= SkyCoord(8.478299999999999,-7.86654138888888, unit=u.deg, frame='icrs')
wcs = image_r.wcs.wcs
square_size_arcsec = 5*u.arcsec
center_x, center_y = skycoord_to_pixel(galloc, wcs)
pixel_scales = proj_plane_pixel_scales(wcs)
scale_x = pixel_scales[0] * u.deg / u.pix
scale_y = pixel_scales[1] * u.deg / u.pix
width_pixels = (square_size_arcsec.to(u.deg) / scale_x).value
height_pixels = (square_size_arcsec.to(u.deg) / scale_y).value
bottom_left_x = center_x - width_pixels / 2
bottom_left_y = center_y - height_pixels / 2
square = patches.Rectangle(
    (bottom_left_x, bottom_left_y),
    width_pixels,
    height_pixels,
    edgecolor='white',
    facecolor='none',
    linewidth=1, zorder=1000)
ax.add_patch(square)


# --- Code to connect the square to the specx axes ---

# 1. Define the coordinates of the square's corners you want to connect.
bottom_right_corner = (bottom_left_x + width_pixels, bottom_left_y)
top_right_corner = (bottom_left_x + width_pixels, bottom_left_y + height_pixels)

# 2. Define the target coordinates on the `specx` axes.
bottom_left_specx = (0, 0) # Bottom-left of specx
top_left_specx = (0, 1)    # Top-left of specx

# 3. Create a ConnectionPatch for the bottom corners.
con1 = patches.ConnectionPatch(
    xyA=bottom_right_corner,   # Point on the square
    xyB=bottom_left_specx,     # Point on the spectrum plot
    coordsA=ax.transData,      # Coordinate system for xyA is `ax`'s data
    coordsB=specx.transAxes,   # Coordinate system for xyB is `specx`'s axes
    color='white',
    lw=1,
)

# 4. Create another ConnectionPatch for the top corners.
con2 = patches.ConnectionPatch(
    xyA=top_right_corner,
    xyB=top_left_specx,
    coordsA=ax.transData,
    coordsB=specx.transAxes,
    color='white',
    lw=1,
)

# 5. Add the connection lines to the FIGURE.
fig.add_artist(con1)
fig.add_artist(con2)

# --- End of connection code ---


pf.fix_plot(axes[1:], tickdir='out')

# To display the plot
plt.show()
