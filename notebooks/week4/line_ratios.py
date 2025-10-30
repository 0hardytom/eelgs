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
    

def get_velocity_disp(fwhm_obs, rest_wavelength, fwhm_inst=2.5): #everything is in angstrom
    c = 299792.458
    fwhm_corr_sq = fwhm_obs**2 - fwhm_inst**2
    if fwhm_corr_sq < 0:
        return np.nan
    fwhm_corr = np.sqrt(fwhm_corr_sq)
    velocity_fwhm = (fwhm_corr / rest_wavelength) * c
    sigma = velocity_fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma #kms-1


def get_metallicity(f_oiii5007, f_oiii4959, f_oiii4363, f_oii3726, f_oii3729, f_hbeta):
    # Input validation
    fluxes = [f_oiii5007, f_oiii4959, f_oiii4363, f_oii3726, f_oii3729, f_hbeta]
    if any(f < 0 for f in fluxes):
        return np.nan
    if f_hbeta == 0:
        return np.nan
    if f_oiii4363 == 0:
        return np.nan
    R_OIII = (f_oiii5007 + f_oiii4959) / f_oiii4363
    if R_OIII <= 7.937:
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


def get_j19(f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta):
    a = -24.135
    b = 6.1532
    c = -0.37866
    d = -0.147
    e = -7.071

    if any(f < 0 for f in [f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta]):
        return np.nan
        
    if f_hbeta == 0:
        return np.nan
    
    oiii_flux_total = f_oiii5007 + f_oiii4959
    oii_flux_total = f_oii3726 + f_oii3729
    
    if oii_flux_total == 0:
        return np.nan

    R23 = (oiii_flux_total + oii_flux_total) / f_hbeta
    logR23 = np.log10(R23)

    O32 = oiii_flux_total / oii_flux_total
    y = np.log10(O32)

    A = c
    B = b - d * y
    C = a - d * e * y - logR23

    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        # No real solution exists for the given line ratios
        print("Warning: No real solution for metallicity (discriminant is negative).")
        return np.nan

    sqrt_discriminant = np.sqrt(discriminant)
    x_upper = (-B + sqrt_discriminant) / (2 * A)
    x_lower = (-B - sqrt_discriminant) / (2 * A)

    if y < 0.5:
        metallicity = x_upper  # Upper branch
    else:  # y >= 0.5
        metallicity = x_lower  # Lower branch

    return metallicity

def _ccm89_k(wave_angstrom):
    if 3030.3 <= wave_angstrom <= 10000: # Optical / NIR
        x = 10000.0 / wave_angstrom # inverse microns
        a = 0.574 * (x**1.61)
        b = -0.527 * (x**1.61)
        # For R_V = 3.1
        k_lambda = a + b / 3.1
        return k_lambda * 3.1
    else:
        raise ValueError("Wavelength is outside the valid range for this simplified CCM89 implementation.")

def calculate_ebv_hbeta_hgamma(f_hbeta, f_hgamma):
    if f_hbeta <= 0 or f_hgamma <= 0:
        return np.nan

    # Theoretical and observed ratios
    intrinsic_ratio = 0.47
    observed_ratio = f_hbeta / f_hgamma

    # Wavelengths in Angstroms
    wave_hbeta = 4861.33
    wave_hgamma = 4340.46

    # Extinction law values
    k_hbeta = _ccm89_k(wave_hbeta)
    k_hgamma = _ccm89_k(wave_hgamma)

    # Calculate E(B-V)
    # Formula derived from: F_obs/F_int = 10^(-0.4 * A_lambda)
    ebv = 2.5 * (np.log10(observed_ratio) - np.log10(intrinsic_ratio)) / (k_hgamma - k_hbeta)
    
    return ebv

def correct_flux_for_extinction(flux, wavelength, ebv):
    if np.isnan(ebv) or flux <= 0:
        return flux # Return original flux if no correction can be applied

    # Get extinction law value for the given wavelength
    k_lambda = _ccm89_k(wavelength)
    
    # Calculate extinction in magnitudes (A_lambda)
    a_lambda = k_lambda * ebv
    
    # Apply correction
    corrected_flux = flux * 10**(0.4 * a_lambda)
    
    return corrected_flux
