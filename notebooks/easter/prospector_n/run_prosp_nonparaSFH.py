import numpy as np
from prospect.io import write_results as writer
from prospect.fitting import fit_model
import sys, os
from astropy.io import ascii
from astropy.table import Table
import helper
from glob import glob
from astropy.constants import c as speedoflight

#------------------------
# Convienence Functions
#------------------------

def tie_gas_logz(logzsol=None, **extras):
    return logzsol

def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

def zfrac_to_masses_log(logmass=None, z_fraction=None, agebins=None, **extras):
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])
    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    if (mass_fraction < 0).any():
        idx = mass_fraction < 0
        if np.isclose(mass_fraction[idx],0,rtol=1e-8):
            mass_fraction[idx] = 0.0
        else:
            raise ValueError('The input z_fractions are returning negative masses!')

    masses = 10**logmass * mass_fraction
    return masses

#----------------------
# SSP function
#-----------------------

def build_sps(**kwargs):
    """
    This is our stellar population model which generates the spectra for stars of a given age and mass. 
    Because we are using a non parametric SFH model, we do have to use a different SPS model than before 
    """
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=1)
    return sps


#--------------------
# Model Setup
#--------------------

def build_model(row,**kwargs):
    from prospect.models import priors, sedmodel
    from astropy.cosmology import Planck18 as cosmo
    print('building model')
    model_params = []
    #basics
    zred = row['z']
    lumdist = cosmo.luminosity_distance(zred).value

    model_params.append({'name': "lumdist", "N": 1, "isfree": False,"init": lumdist,"units": "Mpc"})
    model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2})
    model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 2,'prior': None})
    model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.ClippedNormal(mini=-0.3, maxi=0.3, mean=0.0, sigma=0.3)})
    model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1,'prior': None})
    model_params.append({'name': 'duste_gamma', 'N': 1,'isfree': True,'init': 0.01,'prior': priors.TopHat(mini=0, maxi=1.0)})
    model_params.append({'name': 'duste_umin', 'N': 1,'isfree': True,'init': 1.0,'prior': priors.TopHat(mini=-10, maxi=20.0)})
    model_params.append({'name': 'duste_qpah', 'N': 1,'isfree': True,'init': 3.0,'prior': priors.TopHat(mini=0, maxi=2)})                                                          
    model_params.append({'name': 'add_agb_dust_model', 'N': 1,'isfree': False,'init': 0})
    
    #M-Z
    model_params.append({'name': 'logmass', 'N': 1,'isfree': True,'init': 10,'prior': priors.Uniform(mini=9, maxi=11)})
    model_params.append({'name': 'logzsol', 'N': 1,'isfree': True,'init': -0.5,'prior': priors.Uniform(mini=-2.2, maxi=-1.2)})
    

    #SFH 
    #here, we tell fsps (via Prospector) that we will be using a special SFH (so init=3, which corresponds to a
    #'custom' SFH). Of note is that the "mass" parameter no long refers to the total stellar mass. Instead,
    #this is related to the stellar mass formed in each piece-wise time bin. However, the model doesn't actually
    #sample the mass posteriors. Instead, it uses a proxy variable "z_fraction" that is related to the choice of
    #prior (Dirichlet distribution). If you want to learn more, I'd highly recommend reading Joel Leja's 2019
    #paper introducing the Prospector non parametric SFH models
    model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 3})
    #Now, mass refers to the stellar mass formed *in each time bin* while the logmass parameter above 
    #sets the overall normalization 
    model_params.append({'name': "mass", 'N': 3, 'isfree': False, 'init': 1., 'depends_on':zfrac_to_masses_log})
    #agebins are the limits for each piece-wise bin of star formation. these are set below
    model_params.append({'name': "agebins", 'N': 1, 'isfree': False,'init': []})
    #proxy parameter for SFR in each age bin
    model_params.append({'name': "z_fraction", "N": 2, 'isfree': True, 'init': [0, 0],'prior': priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=5.0)})                                                                                                                                                                                                                           

    #NEBULAR STUFF
    ###### Nebular Emission ###########
    model_params.append({'name': 'add_neb_emission', 'N': 1,
                            'isfree': False,
                            'init': True,
                            'units': r'log Z/Z_\odot',
                            'prior_function_name': None,
                            'prior_args': None})

    model_params.append({'name': 'add_neb_continuum', 'N': 1,
                            'isfree': False,
                            'init': True,
                            'units': r'log Z/Z_\odot',
                            'prior_function_name': None,
                            'prior_args': None})
                            
    model_params.append({'name': 'nebemlineinspec', 'N': 1,
                            'isfree': False,
                            'init': False,
                            'prior_function_name': None,
                            'prior_args': None})

    model_params.append({'name': 'gas_logz', 'N': 1,
                            'isfree': True,
                            'init': 0.0,
                            'depends_on': tie_gas_logz,
                            'units': r'log Z/Z_\odot',
                            'prior': priors.TopHat(mini=-3.0, maxi=1)})

    model_params.append({'name': 'gas_logu', 'N': 1,
                            'isfree': True,
                            'init': -2.0,
                            'units': '',
                            'prior': priors.TopHat(mini=-5.0, maxi=-.5)})


    #here we set the number and location of the timebins, and edit the other SFH parameters to match in size
    n = [p['name'] for p in model_params]
    tuniv = 14. #Gyr, age at z=0                                                                                                                                                                                                         
    nbins=10
    tbinmax = (tuniv * 0.8) * 1e9 #earliest time bin goes from age = 0 to age = 2.8 Gyr
    lim1, lim2 = 7.47, 8.0 #most recent time bins at 30 Myr and 100 Myr ago                                                                                                                                                                                                 
    agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(tuniv*1e9)]
    agebins = np.array([agelims[:-1], agelims[1:]])

    zinit = np.array([(i-1)/float(i) for i in range(nbins, 1, -1)])
    # Set up the prior in `z` variables that corresponds to a dirichlet in sfr
    # fraction. 
    alpha = np.arange(nbins-1, 0, -1)
    zprior = priors.Beta(alpha=alpha, beta=np.ones_like(alpha), mini=0.0, maxi=1.0)

    model_params[n.index('mass')]['N'] = nbins
    model_params[n.index('agebins')]['N'] = nbins
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('z_fraction')]['N'] = nbins-1
    model_params[n.index('z_fraction')]['init'] = zinit
    model_params[n.index('z_fraction')]['prior'] = zprior

    #### now deal with emission lines ####

    model = sedmodel.SedModel(model_params)
    

    return model


#------------------
# Build Observations
#-------------------

def build_obs(row, **kwargs):
    
    from sedpy.observate import load_filters
    from astropy import units as u
    from astropy import constants
    from astropy.coordinates import SkyCoord
    from mpdaf.obj import Cube

    spitzer = ['spitzer_irac_ch'+n for n in ['1','2']]
    filternames = spitzer

    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]

    filters_to_get = ['flux_Spitzer_I1_3.6','flux_Spitzer_I2_4.5']
    flux = np.array([row[header] for header in filters_to_get])*u.uJy
    flux_mag = flux*1e-6/3631
    unc_mag = flux_mag/10 # assuming a SNR here until we calculate errors.

    ## now do spectrum ##
    print('doing spectral stuff!')
    name  = row['name']
    allfiles = glob('/Volumes/Expansion/exp_thardy/cubes/*.fits')+glob('/Volumes/Expansion/exp_thardy/cubes_new/*.fits')
    filedir = 'BLANK'
    for a in allfiles:
        if name in a:
            filedir = a
        
    coord = SkyCoord(ra=row['ra'], dec=row['dec'], unit=u.deg)
    cube = Cube(filedir)
    spectrum = cube.aperture((coord.dec.deg,coord.ra.deg),2)
    spectrum = spectrum.resample(2.5)
    conv_factor = (1.0 / speedoflight).to(u.Jy * u.s * u.cm**2 / (u.erg *u.AA))
    flux_lambda = spectrum.data.data*u.erg/(u.AA*u.s*u.cm**2)*1e-20
    wavelength = spectrum.wave.coord()*u.AA
    flux_jansky = conv_factor * flux_lambda * wavelength**2
    flux_maggie = flux_jansky/3631
    print('spectrum created....')
    
    obs = {}
    #put some useful things in our dictionary. Prospector exepcts to see, at the least, the filters, photmetry
    #and errors, and if available, the spectrum information. I also include the full powderday SED for easy 
    #access later
    obs['filters'] = filters
    obs['maggies'] = flux_mag.value
    obs['maggies_unc'] = unc_mag.value
    obs['phot_mask'] = np.isfinite(flux_mag)
    obs['wavelength'] = wavelength.value
    obs['spectrum'] = flux_maggie.value
    obs['unc'] = flux_maggie.value/10

    return obs

#-------------------
# Put it all together
#-------------------


def build_all(row,**kwargs):

    return (build_obs(row,**kwargs), build_model(row,**kwargs),
            build_sps(**kwargs))


#parameters that will be passed to dynesty, the posterior sampler. typically can just ignore these / use these defaults
run_params = {'verbose':True,
              'debug':False,
              'output_pickles': True,
              'nested_bound': 'multi', # bounding method                                                                                      
              'nested_sample': 'auto', # sampling method                                                                                      
              'nested_nlive_init': 400,
              'nested_nlive_batch': 200,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              }


if __name__ == '__main__':

    PD_DIR ='test_for_sed.csv'
    TAB = Table(ascii.read(PD_DIR))
    for II,ROW in enumerate(TAB):
        obs, model, sps = build_all(ROW,**run_params)
        run_params["sps_libraries"] = sps.ssp.libraries
        run_params["param_file"] = __file__
        hfile = f"test{II}02_fit.h5"
        print('Running fits')
        output = fit_model(obs, model, sps, [None,None],**run_params)
        print('Done. Writing now')
        writer.write_hdf5(hfile, run_params, model, obs,
                output["sampling"][0], output["optimization"][0],
                tsample=output["sampling"][1],
                toptimize=output["optimization"][1])


