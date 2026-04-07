import numpy as np
from prospect.io import write_results as writer
from prospect.fitting import fit_model
import sys, os
from astropy.io import ascii
from astropy.table import Table
from glob import glob
from astropy.constants import c as speedoflight
from prospect.models import priors, sedmodel


#------------------------
# Convienence Functions
#------------------------

def transform_logmass_to_mass(mass=None, logmass=None, **extras):
    return 10**logmass

def load_gp(**extras):
    return None, None

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def transform_zfraction_to_sfrfraction(sfr_fraction=None, z_fraction=None, **extras):
    sfr_fraction[0] = 1-z_fraction[0]
    for i in range(1,sfr_fraction.shape[0]): sfr_fraction[i] =  np.prod(z_fraction[:i])*(1-z_fraction[i])
    return sfr_fraction

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

#############
# MODEL_PARAMS
#############

model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'lumdist', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'Mpc',
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'add_igm_absorption', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'init_disp': 0.25,
                        'disp_floor': 0.2,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-2.2, maxi=-1.2)})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 3,
                        'units': None})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=5.0, maxi=13.0)})

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': False,
                        'init': 1e10,
                        'depends_on': transform_logmass_to_mass,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=1e5, maxi=1e13)})

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'units': 'log(yr)',
                        'prior': None})

model_params.append({'name': 'sfr_fraction', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'depends_on': transform_zfraction_to_sfrfraction,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'z_fraction', 'N': 1,
                        'isfree': True,
                        'init': [],
                        'units': '',
                        'prior': priors.Beta(alpha=1.0, beta=1.0,mini=0.0,maxi=1.0)})

########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                             'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': None,
                             'prior_function_name': None,
                             'prior_args': None})

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'depends_on': to_dust1,
                        'init': 1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=6.0)})

model_params.append({'name': 'dust1_fraction', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.8,
                        'disp_floor': 0.8,
                        'units': '',
                        'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.3,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=3.0)})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.TopHat(mini=-2.2, maxi=0.4)})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-1.5, maxi=-0.5)})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.4,
                        'disp_floor': 0.3,
                        'units': None,
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 10.0,
                        'disp_floor': 5.0,
                        'units': None,
                        'prior': priors.TopHat(mini=-10, maxi=25.0)})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 3.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior': priors.TopHat(mini=0.0, maxi=10.0)})

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
                        'prior': priors.TopHat(mini=-2.0, maxi=0.5)})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': True,
                        'init': -2.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})

####### Calibration ##########
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior': priors.TopHat(mini=-5.0, maxi=0.5)})

####### Units ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mformed'})

#### resort list of parameters 
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','z_fraction', 'dust2', 'logzsol', 'dust_index', 'dust1_fraction', 'duste_qpah', 'duste_gamma', 'duste_umin']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

def build_model(row, agelims =[6.0,7.0,7.5, 8.0, 8.5, 9.0, 0], **kwargs):

    from astropy.cosmology import Planck18 as cosmo
    print('building model')

    #basics
    zred = row['z']
    lumdist = cosmo.luminosity_distance(zred).value

    #### CALCULATE TUNIV #####
    tuniv = cosmo.age(zred).value

    #### NONPARAMETRIC SFH ######
    agelims[-1] = np.log10(tuniv*1e9)
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = len(agelims) - 1

    #### ADJUST MODEL PARAMETERS #####
    n = [p['name'] for p in model_params]

    #### SET UP AGEBINS
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T

    #### FRACTIONAL MASS INITIALIZATION
    # N-1 bins, last is set by x = 1 - np.sum(sfr_fraction)
    model_params[n.index('z_fraction')]['N'] = ncomp-1
    tilde_alpha = np.array([ncomp-i for i in range(1,ncomp)])
    model_params[n.index('z_fraction')]['prior'] = priors.Beta(alpha=tilde_alpha, beta=np.ones_like(tilde_alpha),mini=0.0,maxi=1.0)
    model_params[n.index('z_fraction')]['init'] = np.array([(i-1)/float(i) for i in range(ncomp,1,-1)])
    model_params[n.index('z_fraction')]['init_disp'] = 0.02

    model_params[n.index('sfr_fraction')]['N'] = ncomp-1
    model_params[n.index('sfr_fraction')]['prior'] = priors.TopHat(maxi=np.full(ncomp-1,1.0), mini=np.full(ncomp-1,0.0))
    model_params[n.index('sfr_fraction')]['init'] =  np.zeros(ncomp-1)+1./ncomp
    model_params[n.index('sfr_fraction')]['init_disp'] = 0.02

    #### INSERT REDSHIFT INTO MODEL PARAMETER DICTIONARY ####
    model_params[n.index('zred')]['init'] = zred
    model_params[n.index('lumdist')]['init'] = lumdist

    #### CREATE MODEL
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

    maggie = u.def_unit('maggie', u.Jy/3631)

    spitzer = ['spitzer_irac_ch'+n for n in ['1','2']]
    filternames = spitzer

    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]

    filters_to_get = ['flux_Spitzer_I1_3.6','flux_Spitzer_I2_4.5']
    flux = np.array([row[header] for header in filters_to_get])*u.uJy
    flux_mag = flux*1e-6/3631
    unc_mag = np.abs(flux_mag/10) # assuming a SNR here until we calculate errors.

    ## now do spectrum ##
    print('doing spectral stuff!')
    name  = row['name']
    object_id = row['object_id']
    
    # check if spec folder exists, if not create it
    if not os.path.isdir('spec/'):
        os.mkdir('spec/')
    
    spec_file = f'spec/{object_id}.spec.fits'

    if os.path.exists(spec_file):
        print('loading spectrum from file')
        spec_table = Table.read(spec_file)
        wavelength = spec_table['wavelength'].data * u.AA
        flux_maggie = spec_table['flux_maggie'].data * maggie
    else:
        print('creating spectrum')
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

        # save for later
        spec_table = Table([wavelength.value, flux_maggie.value], names=('wavelength', 'flux_maggie'))
        spec_table.write(spec_file, overwrite=True)

    print('spectrum created....')
    
    obs = {}
    #put some useful things in our dictionary. Prospector exepcts to see, at the least, the filters, photmetry
    #and errors, and if available, the spectrum information. I also include the full powderday SED for easy 
    #access later
    # obs['filters'] = filters
    # obs['maggies'] = flux_mag.value
    # obs['maggies_unc'] = unc_mag.value
    # obs['phot_mask'] = np.isfinite(flux_mag)
    # obs['wavelength'] = wavelength.value
    # obs['spectrum'] = flux_maggie.value
    # snr = 5
    # spec_floor = 0.01 * np.nanmedian(flux_maggie.value)
    # obs['unc'] = np.sqrt((flux_maggie.value / snr)**2 +spec_floor**2)
    # obs['mask'] = np.isfinite(flux_maggie.value)&(flux_maggie.value>0)

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

    PD_DIR ='sedrun.csv'
    TAB = Table(ascii.read(PD_DIR))
    for II,ROW in enumerate(TAB):
        name = ROW['object_id']
        save_string = f'out/{name}'
        if os.path.isdir(save_string):
            os.mkdir(save_string)
        nresults = len(glob(save_string+'/*'))

        obs, model, sps = build_all(ROW,**run_params)
        run_params["sps_libraries"] = sps.ssp.libraries
        run_params["param_file"] = __file__

        hfile = save_string+f'results_{nresults}.h5'
        print('Running fits')
        output = fit_model(obs, model, sps, [None,None],**run_params)
        print('Done. Writing now')
        writer.write_hdf5(hfile, run_params, model, obs,
                output["sampling"][0], output["optimization"][0],
                tsample=output["sampling"][1],
                toptimize=output["optimization"][1])


