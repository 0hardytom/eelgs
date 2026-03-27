import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import sys
import os
import plotfancy as pf
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
from astropy.table import Table
from types import SimpleNamespace
import re
from astropy.io import ascii
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy.modeling import models, fitting

import os
from io import StringIO
from astropy.table import vstack

# from calculate_jiang19_metallicity import calculate_metallicity_jiang19 as cjm19
sys.path.append('../../../')
import src.ifu_tools.line_ratios as lr
# import src.ifu_tools.ifutools as ift
from src.ifu_tools.ifutools import museCube, QT_Candidates
from src.ifu_tools.run_photometry import main as pht_main
# import line_ratios as lr

import logging 
logging.getLogger('mpdaf').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from astropy.table import join
from astroquery.sdss import SDSS

###### RUN SCRIPT #######
pht_main()

print('IR data complete.')