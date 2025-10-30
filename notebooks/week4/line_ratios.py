import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from astropy.coordinates import SkyCoord
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

def get_R23(f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta):
    if any(f < 0 for f in [f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta]):
        print("Error: All input fluxes must be non-negative.")
        return np.nan
    if f_hbeta == 0:
        print("Error: H-beta flux cannot be zero.")
        return np.nan
    oiii_flux_total = f_oiii5007 + f_oiii4959
    oii_flux_total = f_oii3726 + f_oii3729
    R23 = (oiii_flux_total + oii_flux_total) / f_hbeta
    return R23
    

def calculate_velocity_dispersion(fwhm_obs, rest_wavelength, fwhm_inst=2.5): #everything is in angstrom
    c = 299792.458
    fwhm_corr_sq = fwhm_obs**2 - fwhm_inst**2
    if fwhm_corr_sq < 0:
        return np.nan
    fwhm_corr = np.sqrt(fwhm_corr_sq)
    velocity_fwhm = (fwhm_corr / rest_wavelength) * c
    sigma = velocity_fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma #kms-1


def calculate_direct_metallicity(f_oiii5007, f_oiii4959, f_oiii4363, f_oii3726, f_oii3729, f_hbeta):
    # Input validation
    fluxes = [f_oiii5007, f_oiii4959, f_oiii4363, f_oii3726, f_oii3729, f_hbeta]
    if any(f < 0 for f in fluxes):
        print("Error: All input fluxes must be non-negative.")
        return np.nan
    if f_hbeta == 0:
        print("Error: H-beta flux cannot be zero.")
        return np.nan
    if f_oiii4363 == 0:
        print("Error: [OIII] 4363 flux cannot be zero for direct method.")
        return np.nan
    R_OIII = (f_oiii5007 + f_oiii4959) / f_oiii4363
    if R_OIII <= 7.937:
        print(f"Warning: [OIII] ratio ({R_OIII:.2f}) is too low to calculate a positive temperature.")
        return np.nan
    T_e_oiii = 32940 / np.log(R_OIII / 7.937)
    # Campbell, Terlevich & Melnick (1986)
    T_e_oii = 0.7 * T_e_oiii + 3000
    t_e_oiii = T_e_oiii / 10000.0
    t_e_oii = T_e_oii / 10000.0
    # Izotov et al. (2006)
    oiii_flux_total = f_oiii5007 + f_oiii4959
    oii_flux_total = f_oii3726 + f_oii3729
    o_plus_plus_over_h = (oiii_flux_total / f_hbeta) * 1e-6 * (t_e_oiii**0.53) * np.exp(9.8 / t_e_oiii)
    o_plus_over_h = (oii_flux_total / f_hbeta) * 1e-6 * (t_e_oii**0.55) * np.exp(1.96 / t_e_oii)
    o_over_h = o_plus_plus_over_h + o_plus_over_h
    if o_over_h <= 0:
        return np.nan
    return 12 + np.log10(o_over_h)


