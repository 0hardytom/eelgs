import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from astropy.units import u
from astropy.coordinates import SkyCoord
import sys
import plotfancy as pf
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
import argparse

pf.housestyle_rcparams()

from IFU_tools import analyse_galaxy_spectrum

def main():
    parser = argparse.ArgumentParser(description="Analyse galaxy spectrum from a MUSE cube.")
    parser.add_argument('--cube_path', type=str, default='../../cubes/macs0159m34_COMBINED_CUBE_MED_FINAL.fits', help='Path to the MUSE cube')
    parser.add_argument('--title', type=str, default='MACS', help='Title for the analysis')
    parser.add_argument('--radius', type=float, default=0.6, help='Radius in arcseconds')
    parser.add_argument('--Z', type=float, default=0.69, help='Redshift guess')
    parser.add_argument('--ra_hms', type=str, default='01h59m06.05s', help='Right Ascension in hms format')
    parser.add_argument('--dec_dms', type=str, default='-34d13m03.8s', help='Declination in dms format')

    args = parser.parse_args()

    coords = SkyCoord(args.ra_hms, args.dec_dms, frame='icrs')
    RA_DEG = coords.ra.deg
    DEC_DEG = coords.dec.deg

    spec, lf1 = analyse_galaxy_spectrum(
        cube_path=args.cube_path,
        ra=RA_DEG,
        dec=DEC_DEG,
        radius=args.radius,
        title=args.title,
        z_guess=args.Z,
        pref=args.ra_hms.replace('.','p')+'_'+args.dec_dms.replace('.','p')
    )

if __name__ == '__main__':
    main()