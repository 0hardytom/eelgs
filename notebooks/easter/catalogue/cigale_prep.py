import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from astropy.coordinates import SkyCoord
import sys
import os
import plotfancy as pf
from matplotlib.patches import Circle, ConnectionPatch, FancyArrowPatch, Arrow
from astropy.visualization import ZScaleInterval

from types import SimpleNamespace
import re
from astropy.io import ascii, fits
from astropy import units as u
from astropy.constants import c as speedoflight
from astropy.table import Table, vstack, hstack, join
from scipy.optimize import curve_fit, root
from astropy.cosmology import Planck18 as cosmo
from astropy import coordinates as coords
from astroquery.sdss import SDSS
from requests.exceptions import ConnectionError
from matplotlib.lines import Line2D
# from hst_phot import *
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
import prospect.fitting as fitting
from prospect.io import write_results as writer
import prospect.io.read_results as reader
from prospect.sources import CSPSpecBasis
from prospect.models import priors
from genesis_metallicity.genesis_metallicity import genesis_metallicity
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

# from calculate_jiang19_metallicity import calculate_metallicity_jiang19 as cjm19
sys.path.append('../../../')
import src.ifu_tools.line_ratios as lr
import src.ifu_tools.ifutools as ift

import logging 
logging.getLogger('mpdaf').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
lambda_to_spectral_key = {
    'oiii5007_flux': 'line.OIII-500.7',
    'oiii4959_flux': 'line.OIII-495.9',
    'oii3726_flux':  'line.OII-372.6',
    'oii3729_flux':  'line.OII-372.9',

    'halpha_flux':   'line.H-alpha',
    'hbeta_flux':    'line.H-beta',
    'hgamma_flux':   'line.H-gamma',
    'hdelta_flux':   'line.H-delta',
    'hepsilon_flux': 'line.H-epsilon',

    'oiii4363_flux': 'line.OIII-436.3',
    'neiii_flux':    'line.NeIII-386.9',

    'nii6583_flux':  'line.NII-658.3',
    'nii6548_flux':  'line.NII-654.8',
    'sii6716_flux':  'line.SII-671.6',
    'sii6731_flux':  'line.SII-673.1',
    'nev3426_flux':  'line.NeV-342.6',

    'hei5876_flux':  'line.HeI-587.5',
}

filter_key_to_band = {
    # "flux_HST_F435W":       "hst.acs.wfc.F435W",
    # "flux_HST_F606W":       "hst.acs.wfc.F606W",
    # "flux_HST_F814W":       "hst.acs.wfc.F814W",
    # "flux_HST_F125W":       "hst.wfc3.ir.F125W",
    # "flux_HST_F160W":       "hst.wfc3.ir.F160W",
    "flux_Spitzer_I1_3.6":  "spitzer.irac.I1",
    "flux_Spitzer_I2_4.5":  "spitzer.irac.I2",
}

def main():
    redshift = {'z':'redshift', 'object_id':'id'}

    master_key = filter_key_to_band | lambda_to_spectral_key | redshift
    master_key_inverse = dict((v, k) for k, v in master_key.items())

    master = Table(ascii.read('MASTER.csv'))

    column_names = ['id','redshift']
    column_names.extend(lambda_to_spectral_key.values())
    column_names.extend(filter_key_to_band.values())
    blank = np.zeros([len(column_names), len(master['object_id'])], dtype=np.float64)*np.nan
    cigaletable = Table(blank.T, names=column_names)

    for c in column_names:

        if c in list(lambda_to_spectral_key.values()):
            cigaletable[c] = master[master_key_inverse.get(c)]*1e-23*(u.W / u.m**2)
            cigaletable[c+'_err'] = master[master_key_inverse.get(c)+'_err']*1e-23*(u.W / u.m**2)
        elif c in list(filter_key_to_band.values()):
            cigaletable[c] = master[master_key_inverse.get(c)]*1e-3*u.mJy
        else:
            cigaletable[c] = master[master_key_inverse.get(c)]

    for c in list(cigaletable.columns)[1:]:
        mask1 = (cigaletable[c]*1e30)<=0.0
        cigaletable[c][mask1] = np.nan

    cigaletable.write('/Users/thardy/Documents/durham/eelgs/notebooks/easter/cigale/cigale/sedpeas.csv', overwrite=True)

    props = list(cigaletable.columns[2:])
    print('Copy the following into the ini file:')
    stri = ''
    for p in props:
        stri = stri+p+', '

    print(stri)

if __name__ == "__main__":
    main()