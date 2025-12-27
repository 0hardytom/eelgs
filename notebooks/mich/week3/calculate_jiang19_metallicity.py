import numpy as np

def calculate_metallicity_jiang19(f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta):
    a = -24.135
    b = 6.1532
    c = -0.37866
    d = -0.147
    e = -7.071

    if any(f < 0 for f in [f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta]):
        print("Error: All input fluxes must be non-negative.")
        return np.nan
        
    if f_hbeta == 0:
        print("Error: H-beta flux cannot be zero.")
        return np.nan
    
    oiii_flux_total = f_oiii5007 + f_oiii4959
    oii_flux_total = f_oii3726 + f_oii3729
    
    if oii_flux_total == 0:
        print("Error: Total [OII] flux is zero, cannot calculate O32 ratio.")
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

