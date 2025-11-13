import numpy as np

def kewley01(log_nii_ha):
    """
    Calculates the theoretical "maximum starburst" line from Kewley et al. (2001).
    
    This demarcation line separates AGN from star-forming/composite galaxies. 
    Points above this line are generally classified as AGN.

    Parameters
    ----------
    log_nii_ha : float or array-like
        The log10 of the [N II]/H-alpha flux ratio.

    Returns
    -------
    float or array-like
        The corresponding log10([O III]/H-beta) value for the demarcation line.
    """
    return 0.61 / (log_nii_ha - 0.47) + 1.19

def kauffmann03(log_nii_ha):
    """
    Calculates the empirical demarcation line from Kauffmann et al. (2003).

    This line separates purely star-forming galaxies from composite galaxies.
    Points below this line are classified as star-forming.

    Parameters
    ----------
    log_nii_ha : float or array-like
        The log10 of the [N II]/H-alpha flux ratio.

    Returns
    -------
    float or array-like
        The corresponding log10([O III]/H-beta) value for the demarcation line.
    """
    return 0.61 / (log_nii_ha - 0.05) + 1.3

def classify_bpt(log_nii_ha, log_oiii_hb):
    """
    Classifies galaxies into Star-forming, Composite, and AGN based on BPT diagram position.

    Parameters
    ----------
    log_nii_ha : array-like
        The log10 of the [N II]/H-alpha flux ratio (x-axis values).
    log_oiii_hb : array-like
        The log10 of the [O III]/H-beta flux ratio (y-axis values).

    Returns
    -------
    dict
        A dictionary containing boolean masks for each classification:
        'starburst': Star-forming galaxies
        'transition': Composite/transition objects
        'agn': Active Galactic Nuclei
    """
    # Calculate the y-values of the demarcation lines for each galaxy's x-value
    y_kauffmann = kauffmann03(log_nii_ha)
    y_kewley = kewley01(log_nii_ha)

    # --- Classification Conditions ---
    # 1. Below the Kauffmann line is Star-forming ('starburst')
    #    (only defined for log_nii_ha < 0.05)
    is_starburst = (log_oiii_hb < y_kauffmann) & (log_nii_ha < 0.05)

    # 2. Above the Kewley line is AGN
    #    (also includes objects with log_nii_ha > 0.47, where the line is not defined)
    is_agn = (log_oiii_hb > y_kewley) | (log_nii_ha >= 0.47)

    # 3. Everything in between is a Composite/Transition object
    #    We find this by selecting objects that are NOT starburst and NOT AGN.
    is_transition = (~is_starburst) & (~is_agn)

    return {
        'starburst': is_starburst,
        'transition': is_transition,
        'agn': is_agn
    }


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
        # print('WARNING: Negative Flux')
        return np.nan
    if f_hbeta == 0:
        print('WARNING: No Hbeta')
        return np.nan
    if f_oiii4363 == 0:
        print('WARNING: No oiii4363')
        return np.nan
    R_OIII = (f_oiii5007 + f_oiii4959) / f_oiii4363
    if R_OIII <= 7.937:
        # print('WARNING: R_OIII too small')
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
        # print('WARNING: o_over_h<0')
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

    discriminant = (b-d*y)**2 - 4*c*(a-d*e*y-logR23)
    if discriminant<0:
        return (d*y-b)/(2*c)
    else:
        if y>0.5: #includes the >0.6 and the upper branch - applies for most galaxies
            return ((d*y-b)-np.sqrt(discriminant))/(2*c)
        else: #anything <0.5
            return ((d*y-b)+np.sqrt(discriminant))/(2*c)


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

def get_ebv(observed_ratio, intrinsic_ratio = 0.47, wave_one =4861.33, wave_two=4340.46 ):

    # Extinction law values
    k_one = _ccm89_k(wave_one)
    k_two = _ccm89_k(wave_two)

    # Calculate E(B-V)
    # Formula derived from: F_obs/F_int = 10^(-0.4 * A_lambda)
    ebv = 2.5 * (np.log10(observed_ratio) - np.log10(intrinsic_ratio)) / (k_two - k_one)
    
    return ebv

def correct_flux(flux, wavelength, ebv):
    if np.isnan(ebv) or flux <= 0:
        return flux # Return original flux if no correction can be applied

    # Get extinction law value for the given wavelength
    k_lambda = _ccm89_k(wavelength)
    
    # Calculate extinction in magnitudes (A_lambda)
    a_lambda = k_lambda * ebv
    
    # Apply correction
    corrected_flux = flux * 10**(0.4 * a_lambda)
    
    return corrected_flux


def get_R23_with_errors(f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta,
                        err_f_oiii5007, err_f_oiii4959, err_f_oii3726, err_f_oii3729, err_f_hbeta):
    
    if any(f < 0 for f in [f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta]):
        return np.nan, np.nan
    if f_hbeta == 0:
        return np.nan, np.nan

    oiii_flux_total = f_oiii5007 + f_oiii4959
    oii_flux_total = f_oii3726 + f_oii3729
    
    err_oiii_flux_total_sq = err_f_oiii5007**2 + err_f_oiii4959**2
    err_oii_flux_total_sq = err_f_oii3726**2 + err_f_oii3729**2
    
    numerator = oiii_flux_total + oii_flux_total
    err_numerator_sq = err_oiii_flux_total_sq + err_oii_flux_total_sq
    
    R23 = numerator / f_hbeta
    
    if R23 == 0 or numerator == 0:
        return R23, np.nan

    err_R23 = R23 * np.sqrt((err_numerator_sq / (numerator**2)) + (err_f_hbeta**2 / (f_hbeta**2)))
    
    return R23, err_R23

def get_metallicity_with_errors(f_oiii5007, f_oiii4959, f_oiii4363, f_oii3726, f_oii3729, f_hbeta,
                                err_f_oiii5007, err_f_oiii4959, err_f_oiii4363, err_f_oii3726, err_f_oii3729, err_f_hbeta,
                                n_mc=1000):

    fluxes = np.array([f_oiii5007, f_oiii4959, f_oiii4363, f_oii3726, f_oii3729, f_hbeta])
    errors = np.array([err_f_oiii5007, err_f_oiii4959, err_f_oiii4363, err_f_oii3726, err_f_oii3729, err_f_hbeta])

    if np.any(fluxes < 0):
        return np.nan, np.nan

    rand_fluxes = np.random.normal(loc=fluxes, scale=errors, size=(n_mc, len(fluxes)))

    metallicities = []
    for f in rand_fluxes:
        metallicity = get_metallicity(f[0], f[1], f[2], f[3], f[4], f[5])
        if not np.isnan(metallicity):
            metallicities.append(metallicity)
    
    if not metallicities:
        # print('WARNING: metallicites empty')
        return np.nan, np.nan

    mean_metallicity = np.mean(metallicities)
    ste_metallicity = np.std(metallicities)/np.sqrt(len(metallicities))
    
    return mean_metallicity, ste_metallicity

def get_j19_with_errors(f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta,
                        err_f_oiii5007, err_f_oiii4959, err_f_oii3726, err_f_oii3729, err_f_hbeta,
                        n_mc=1000):

    fluxes = np.array([f_oiii5007, f_oiii4959, f_oii3726, f_oii3729, f_hbeta])
    errors = np.array([err_f_oiii5007, err_f_oiii4959, err_f_oii3726, err_f_oii3729, err_f_hbeta])

    if np.any(fluxes < 0):
        return np.nan, np.nan

    rand_fluxes = np.random.normal(loc=fluxes, scale=errors, size=(n_mc, len(fluxes)))

    j19_values = []
    for f in rand_fluxes:
        j19 = get_j19(f[0], f[1], f[2], f[3], f[4])
        if not np.isnan(j19):
            j19_values.append(j19)
            
    if not j19_values:
        return np.nan, np.nan

    mean_j19 = np.mean(j19_values)
    ste_j19 = np.std(j19_values)/np.sqrt(len(j19_values))
    
    return mean_j19, ste_j19