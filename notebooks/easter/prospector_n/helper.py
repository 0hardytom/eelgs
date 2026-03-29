import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.constants import L_sun
from astropy.table import Row

rest_lambdas = {
        # --- Primary [OIII] and [OII] ---
        'oiii5007': 5006.84,
        'oiii4959': 4958.91,
        'oii3726':  3726.03,
        'oii3729':  3728.82,

        # --- Hydrogen Balmer Series ---
        'halpha':   6562.80,
        'hbeta':    4861.33,
        'hgamma':   4340.46,
        'hdelta':   4101.73,
        'hepsilon': 3970.08,
        'hzeta':    3889.06,
        'heta':     3835.40,

        # --- Key Diagnostic Lines ---
        'oiii4363': 4363.21,
        'neiii':    3868.75,

        # --- Low-Ionization Lines ---
        'nii6583':  6583.45,
        'nii6548':  6548.05,
        'sii6716':  6716.44,
        'sii6731':  6730.82,
        'nev3426': 3426.00,
        'fevii3760': 3760.00,

        # --- Helium Lines ---
        'heii4686': 4685.68,
        'hei5876':  5875.62,
        }

def get_emission_line_luminosity(flux, redshift):
    luminosity_distance = cosmo.luminosity_distance(redshift)
    # Convert the flux to erg/(s cm^2) with astropy units
    flux_cgs = flux * 1e-20 * u.erg / (u.s * u.cm**2)
    # Calculate the luminosity in erg/s
    luminosity_ergs = (4 * np.pi * luminosity_distance**2 * flux_cgs).to(u.erg/u.s)
    # Convert to solar luminosities
    luminosity_solar = luminosity_ergs / L_sun.to(u.erg/u.s)
    return luminosity_solar.value

def emline_io(row:Row, key:str):
    emline_f = row[string+'_flux']
    gal_z = row['z']
    line_AA = rest_lambdas[key]
    emline_L = get_emission_line_luminosity(emline_f, gal_z)
    return line_AA, emline_L
