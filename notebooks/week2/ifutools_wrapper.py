import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from astropy.units import u
from astropy.coordinates import SkyCoord
import sys
import plotfancy as pf
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval

pf.housestyle_rcparams()

from IFU_tools import analyse_galaxy_spectrum

CUBE_PATH = '../../cubes/macs0159m34_COMBINED_CUBE_MED_FINAL.fits'
# CUBE_PATH = '../../cubes/s780_COMBINED_CUBE_MED_FINAL.fits'
TITLE = 'MACS'
RADIUS_ARCSEC = .6
Z = 0.69

ra_hms = '01h59m06.05s'
dec_dms = '-34d13m03.8s'
coords = SkyCoord(ra_hms, dec_dms, frame='icrs')
RA_DEG = coords.ra.deg
DEC_DEG = coords.dec.deg

spec, lf1 = analyse_galaxy_spectrum(
    cube_path=CUBE_PATH,
    ra=RA_DEG,
    dec=DEC_DEG,
    radius=RADIUS_ARCSEC,
    title=TITLE,
    z_guess=Z,
    pref=ra_hms.replace('.','p')+'_'+dec_dms.replace('.','p')
)