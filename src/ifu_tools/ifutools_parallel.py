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
from astropy.io import ascii, fits
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy.modeling import models, fitting
import concurrent.futures


# from calculate_jiang19_metallicity import calculate_metallicity_jiang19 as cjm19
sys.path.append('../../')
import src.ifu_tools.line_ratios as lr
# import line_ratios as lr

import logging 
logging.getLogger('mpdaf').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from astropy.table import join
from astroquery.sdss import SDSS

pf.housestyle_rcparams()

class museCube:
    def __init__(self, path:str, cluster_ra:np.float64, cluster_dec:np.float64, loud=False):
        ### attributes ###
        self.loud = loud
        self.centre = SkyCoord(cluster_ra*u.deg, cluster_dec*u.deg,frame='icrs')
        if self.loud:
            print(self.centre.ra.deg, self.centre.dec.deg)
        self.path = path
        self.title = self._get_title()
        self.spectra = {}
        self.rest_spectra = {}
        ### initialisers ###
        self.init_cube(self.path)
        self.init_lambda()
        self.init_table()

    ### INITIALISERS ###
    def _dirmanagement(self,id):
        if not os.path.isdir(f'figs/{self.title}/{id}/balmer'):
            if not os.path.isdir(f'figs/{self.title}/{id}'):
                if not os.path.isdir(f'figs/{self.title}'):
                    if not os.path.isdir('figs'):
                        os.mkdir(f'figs')
                    os.mkdir(f'figs/{self.title}')
                os.mkdir(f'figs/{self.title}/{id}')
                os.mkdir(f'figs/{self.title}/{id}/lines')
            os.mkdir(f'figs/{self.title}/{id}/balmer')
        if not os.path.isdir(f'dat'):
            os.mkdir(f'dat')
        if not os.path.isdir(f'spec/{self.title}'):
            if not os.path.isdir('spec'): 
                os.mkdir('spec')  
            os.mkdir(f'spec/{self.title}')    
        return True

    def _get_title(self):
        pattern = r'cubes/([^_]+)_'
        match = re.search(pattern, self.path)
        if match:
            return match.group(1)
        else:
            return 'BLANK'

    def init_cube(self, PATH):
        if self.loud: print(f'Loading datacube: {PATH}...')
        try:
            cube = Cube(PATH)
            if self.loud: print(f'Successfully loaded cube. Dimensions: {cube.shape}')
            self.cube = cube
            return True
        except FileNotFoundError:
            print(f'Error: Datacube not found at {PATH}')
            sys.exit(1)
    
    def init_lambda(self):
        self.rest_lambdas = {
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
        self.balmer_lambda = {
        # 'halpha':   6562.80,
        'hbeta':    4861.33,
        'hgamma':   4340.46,
        'hdelta':   4101.73,
        'hepsilon': 3970.08,
        'hzeta':    3889.06,
        'heta':     3835.40,
        }
        self.lambda_keys = {
        # --- Primary [OIII] and [OII] ---
        'oiii5007': r'[OIII] $\lambda$5007',
        'oiii4959': r'[OIII] $\lambda$4959',
        'oii3726':  r'[OII] $\lambda$3726',
        'oii3729':  r'[OII] $\lambda$3729',

        # --- Hydrogen Balmer Series ---
        'halpha':   r'H$\alpha$',
        'hbeta':    r'H$\beta$',
        'hgamma':   r'H$\gamma$',
        'hdelta':   r'H$\delta$',
        'hepsilon': r'H$\epsilon$',
        'hzeta':    r'H$\zeta$',
        'heta':     r'H$\eta$',

        # --- Key Diagnostic Lines ---
        'oiii4363': r'[OIII] $\lambda$4363',  # Auroral line
        'neiii':    r'[NeIII] $\lambda$3869',

        # --- Low-Ionization Lines ---
        'nii6583':  r'[NII] $\lambda$6583',
        'nii6548':  r'[NII] $\lambda$6548',
        'sii6716':  r'[SII] $\lambda$6716',
        'sii6731':  r'[SII] $\lambda$6731',
        'nev3426':  r'[NeV] $\lambda$3426',
        'fevii3760':r'[FeVII] $\lambda$3760',

        # --- Helium Lines (not forbidden) ---
        'heii4686': r'HeII $\lambda$4686',
        'hei5876':  r'HeI $\lambda$5876',
        }
        return True
    
    def init_table(self, cnames=None):
        self.top = ['object_id', 'ra', 'dec','z','angdisp',
                    'foreground', 'cluster_member', 'lensed',
                    'Z_dir','Z_dir_e',
                    'Z_j19','Z_j19_e',
                    'R23','R23_e',
                    'mean_vel_disp','sterr_vel_disp',
                    'zcluster','name'] 
        self.column_names = self.top if cnames==None else cnames
        self.meta = ['flux','flux_err', 'ew','ew_err', 'centroid', 'fwhm','vel_disp']
        for key, _ in self.rest_lambdas.items():
            for m in self.meta:
                self.column_names.append(key+'_'+m)

        dtypes = []
        for name in self.column_names:
            if name in ['object_id', 'name']:
                dtypes.append(str)
            elif name in ['foreground', 'cluster_member', 'lensed']:
                dtypes.append(int)
            else:
                dtypes.append(np.float64)
        
        self.ex_table = Table(names=self.column_names,dtype=dtypes)

        # self.ex_table = Table(names=self.column_names,dtype=([str]+(len(self.column_names)-1)*[np.float64]))
        self.ex_table.add_index('object_id')
        return True
    
    ### BASIC METHODS ###
    def extract_region_centres(self,reg_file_path): #for reading ds9 reg files
        coord_pattern = re.compile(r'\w+\(([^,]+),([^,]+),.*\)')
        centres = []
        try:
            with open(reg_file_path, 'r') as f:
                for line in f:
                    line_strip = line.strip()
                    if line_strip.startswith('#') or line_strip in ['global', 'fk5', 'icrs', 'image']:
                        continue
                    match = coord_pattern.match(line_strip)
                    if match:
                        ra_str, dec_str = match.groups()
                        try:
                            ra = float(ra_str)
                            dec = float(dec_str)
                            centres.append((ra, dec))
                        except ValueError:
                            print(f"Warning: Could not parse coordinates from line: {line_strip}")
        except FileNotFoundError:
            print(f"Error: Region file not found at {reg_file_path}")
            return None
        return centres
    
    def write_regions_to_csv(self,reg_file_path, csv_file_path):
        coordinates = self.extract_region_centres(reg_file_path)
        if coordinates is None:
            print("Could not extract coordinates. Aborting CSV write.")
            return False
        try:
            coord_table = Table(rows=coordinates, names=('ra', 'dec'))
            coord_table.write(csv_file_path, format='csv', overwrite=True)
            return True
        except Exception as e:
            print(f"An error occurred while writing with Astropy: {e}")
            return False

    # def extract_spectrum(self, ra, dec, radius):
    #     centre = (dec, ra)
    #     spec = self.cube.aperture(centre, radius, is_sum=True)
    #     return spec

    def extract_spectrum(self, ra, dec, radius, redshift, corrRADEC = False):
        if not corrRADEC:
            centre = (dec, ra)
            print(centre)
            spec = self.cube.aperture(centre, radius, is_sum=True)
            return spec      

        else:
            OIII_REST_WAVE = 5007.0
            oiii_observed_wave = OIII_REST_WAVE * (1 + redshift)
            wave_window = 20.0
            lambda_min = oiii_observed_wave - (wave_window / 2)
            lambda_max = oiii_observed_wave + (wave_window / 2)

            subcube = self.cube.subcube((dec,ra),1,(lambda_min,lambda_max))

            oiii_image = subcube.sum(axis=0)

            peak_info = oiii_image.peak()
            corrected_dec = peak_info['y']
            corrected_ra = peak_info['x']

            print(type(corrected_dec))
            print(dec, ra)
            print(corrected_dec,corrected_ra)

            corrected_centre = (corrected_dec, corrected_ra)
            spec = self.cube.aperture(corrected_centre, radius, is_sum=True)

            return spec, corrected_ra, corrected_dec
    
    def deredshift_spectrum(self, spec, z):
        spec_rest = spec.copy()
        spec_rest.wave.set_crval(spec_rest.wave.get_crval() / (1 + z))
        spec_rest.wave.set_step(spec_rest.wave.get_step() / (1 + z))
        return spec_rest
    
    def generate_gaussian_profile(self, params, num_points=500, width_factor=4.0):
        sigma = params.fwhm / (2 * np.sqrt(2 * np.log(2)))
        x_min = params.lpeak - width_factor * params.fwhm / 2
        x_max = params.lpeak + width_factor * params.fwhm / 2
        x = np.linspace(x_min, x_max, num_points)
        y = params.cont + params.peak * np.exp(-((x - params.lpeak) ** 2) / (2 * sigma ** 2))
        return np.vstack((x, y))
    
    def find_z_from_line(self, spec, zapprox, target_str:str='oiii5007'):
        target = self.rest_lambdas[target_str]
        obs_wave_guess = target * (1 + zapprox)
        fit = spec.gauss_fit(lmin=(obs_wave_guess - 20), lmax=(obs_wave_guess + 20), plot=False)
        line_z = (fit.lpeak / target) - 1
        return line_z
    
    def makeid(self, cds):
        return str(cds.ra).replace('.','pt') + str(cds.dec).replace('.','pt')
    
    def write_table(self):
        self.ex_table.write(f'dat/{self.title}_data.csv', overwrite=True)
    
    ### PLOTTING FUNCTIONS ###
    def plot_spectrum_and_cutout(self,spec, ra, dec, id):
        title = self.title
        fig, ax = pf.create_plot(size=(8, 2))
        ax_cont = fig.add_axes((1.02, 0, 1/4, 1))
        spec.plot(ax=ax, title=fr'{title} at (RA={ra:.4f}, Dec={dec:.4f})$^\circ$', color='#ff004f')
        ax.set_xlabel(r'Wavelength, $\lambda$, [$\AA$]')
        ax.set_ylabel(r'Flux [$\times10^{-20}\,\mathrm{erg}/\AA\,s\,\mathrm{cm}^{-2}$]')

        subcube_cont = self.cube.select_lambda(7000, 7500)
        im_cont = subcube_cont.mean(axis=0)
        im_cutout = im_cont.subimage(center=(dec, ra), size=4.0)

        vmin, vmax = ZScaleInterval().get_limits(im_cutout.data)
        im_cutout.plot(ax=ax_cont, vmin=vmin, vmax=vmax, show_xlabel=False, show_ylabel=False, cmap='magma')
        ax_cont.set_xticks([])
        ax_cont.set_yticks([])

        centre_pix_yx = im_cutout.wcs.sky2pix([dec, ra], 1)[0]
        aperture_circle = Circle((centre_pix_yx[1], centre_pix_yx[0]), 3, edgecolor='white', facecolor='none', lw=2, zorder=10)
        ax_cont.add_patch(aperture_circle)

        pf.fix_plot([ax])
        fig.savefig(f'figs/{title}/{id}/spectrum.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        return True

    def plot_extracted_line(self, rest_spec, fit_params, target_str, id):
        profile = self.generate_gaussian_profile(fit_params, 500, 100)

        fig, ax = pf.create_plot(size=(4,2))

        ax.set_title('Line Fit for '+self.lambda_keys[target_str])
        
        rest_spec.plot(ax=ax, color='#ff004f')
        ax.plot(profile[0], profile[1], color='k', lw=1.4)

        ax.set_xlim(np.min(profile[0]), np.max(profile[0]))
        ax.set_ylim(-0.1*np.max(profile[1]), 1.5*np.max(profile[1]))

        ax.set_xlabel(r'O$_{III}$-Calibrated Rest-$\lambda$, [$\AA$] }')
        ax.set_ylabel(r'Flux [$\times10^{-20}\,\mathrm{erg}/\AA\,s\,\mathrm{cm}^{-2}$]')

        pf.fix_plot([ax])
        fig.savefig(f'figs/{self.title}/{id}/lines/spectrum_{target_str}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def plot_all_lines_on_spectrum(self, rest_spec, linefits, id, ra=None, dec=None):
        fig, ax = pf.create_plot(size=(8, 2))
        if ra is not None and dec is not None:
            ax_cont = fig.add_axes((1.02, 0, 1/4, 1))

        rest_spec.plot(ax=ax, color='#ff004f', label='Spectrum')

        line_names = list(self.rest_lambdas.keys())
        
        max_peak_y = np.max(rest_spec.data)

        for i, linefit in enumerate(linefits):
            if linefit.flux != 0:
                profile = self.generate_gaussian_profile(linefit, 500, 4)
                # if not np.sum(profile[1]<0)>0:
                if linefit.peak>0:
                    ax.plot(profile[0], profile[1], color='k', lw=1.2)
                    
                    line_name = line_names[i]
                    label = self.lambda_keys[line_name]
                    
                    peak_y = linefit.cont + linefit.peak
                    if peak_y > max_peak_y:
                        max_peak_y = peak_y
                    ax.text(linefit.lpeak, peak_y * 1.05, label, ha='center', va='bottom', rotation=90, fontsize=8)
    
        ax.set_xlabel(r'Rest Wavelength, $\lambda$, [$\AA$]')
        ax.set_ylabel(r'Flux [$\times10^{-20}\,\mathrm{erg}/\AA\,s\,\mathrm{cm}^{-2}$]')

        if max_peak_y > 0:
            ax.set_ylim(bottom=-50, top=max_peak_y * 1.5)
        else:
            ax.set_ylim(bottom=-50)

        # Continuum plot part
        if ra is not None and dec is not None:
            subcube_cont = self.cube.select_lambda(7000, 7500)
            im_cont = subcube_cont.mean(axis=0)
            im_cutout = im_cont.subimage(center=(dec, ra), size=4.0)

            vmin, vmax = ZScaleInterval().get_limits(im_cutout.data)
            im_cutout.plot(ax=ax_cont, vmin=vmin, vmax=vmax, show_xlabel=False, show_ylabel=False, cmap='magma')
            ax_cont.set_xticks([])
            ax_cont.set_yticks([])

            centre_pix_yx = im_cutout.wcs.sky2pix([dec, ra], 1)[0]
            aperture_circle = Circle((centre_pix_yx[1], centre_pix_yx[0]), 3, edgecolor='white', facecolor='none', lw=2, zorder=10)
            ax_cont.add_patch(aperture_circle)

        pf.fix_plot([ax])
        fig.savefig(f'figs/{self.title}/{id}/spectrum_all_lines.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        return True
    
    def balmer_diagnostic_plot(self, rest_spec, linefits, id):

        fig, ax = pf.create_plot(size=(8, 2))

        rest_spec.plot(ax=ax, color='#ff004f', label='Spectrum')

        line_names = list(self.balmer_lambda.keys())
        fluxes = []
        
        max_peak_y = np.max(rest_spec.data)

        for i, linefit in enumerate(linefits):
            fluxes.append(linefit.flux)
            if linefit.flux != 0:
                profile = self.generate_gaussian_profile(linefit, 500, 4)
                # if not np.sum(profile[1]<0)>0:
                if linefit.peak>0:
                    ax.plot(profile[0], profile[1], color='k', lw=1.2)
                    
                    line_name = line_names[i]
                    label = self.lambda_keys[line_name]
                    
                    peak_y = linefit.cont + linefit.peak
                    if peak_y > max_peak_y:
                        max_peak_y = peak_y
                    ax.text(linefit.lpeak, peak_y * 1.05, label, ha='center', va='bottom', rotation=90, fontsize=8)

        fluxes = np.array(fluxes)
        fluxes_norm = fluxes/fluxes[0]
        flux_string = ":".join([f"{f:.2f}" for f in fluxes_norm[:5]])
        ax.set_title(r'$\beta:\gamma:\delta:\epsilon:\zeta:\nu=$'+flux_string)
        ax.set_xlabel(r'Rest Wavelength, $\lambda$, [$\AA$]')
        ax.set_ylabel(r'Flux [$\times10^{-20}\,\mathrm{erg}/\AA\,s\,\mathrm{cm}^{-2}$]')

        ax.set_ylim(bottom=-0.11*max_peak_y, top=max_peak_y * 1.5)

        ax.fill_between([0,3646], 2*[-1000], 2*[4*max_peak_y], color='#77aca2', zorder=-10, alpha=0.3)
        ax.fill_between([3646,10000], 2*[-1000], 2*[4*max_peak_y], color='#ff004f', zorder=-10, alpha=0.3)

        ax.text(3100,1.2*max_peak_y,'Continuum')
        ax.text(4600,1.2*max_peak_y,'Series')

        pf.fix_plot([ax])
        fig.savefig(f'figs/{self.title}/{id}/balmer/balmer_diag.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        return True


    ### EXTRACTION METHODS ###

    # def fit_line(self,dered_spectra, target_str:str):
    #     target = self.rest_lambdas[target_str]
    #     try:
    #         line_fit = dered_spectra.gauss_fit(lmin=(target - 50), lmax=(target + 50),lpeak=target ,plot=False)

    #         # check for the validity of the fit
    #         sub_spec = dered_spectra.subspec(lmin=target - 15, lmax=target + 15)
    #         if sub_spec is not None and line_fit is not None and hasattr(line_fit, 'peak') and hasattr(line_fit, 'cont'):
    #             max_data_value = np.max(sub_spec.data)
    #             fitted_peak_value = line_fit.peak + line_fit.cont
    #             if fitted_peak_value > 1.5 * max_data_value:
    #                 raise ValueError("Fitted peak is unrealistically high.")
    #     except Exception:
    #         sub_spec = dered_spectra.subspec(lmin=target - 10, lmax=target + 10)
            
    #         spec_for_continuum = None
    #         if sub_spec is not None and len(sub_spec.shape) == 1 and sub_spec.shape[0] > 2:
    #             spec_for_continuum = sub_spec
    #         elif dered_spectra is not None and len(dered_spectra.shape) == 1 and dered_spectra.shape[0] > 2:
    #             spec_for_continuum = dered_spectra
    #         if spec_for_continuum:
    #             continuum = np.mean(spec_for_continuum.data)
    #             std_err = np.std(spec_for_continuum.data)
                
    #             fit_mock = SimpleNamespace(
    #                 flux=0.0,
    #                 err_flux=std_err,
    #                 peak=0.0,
    #                 cont=continuum,
    #                 lpeak=target,
    #                 fwhm=1.0,
    #                 err_peak=std_err,
    #                 err_cont=std_err,
    #                 err_lpeak=0.0,
    #                 err_fwhm=0.0
    #             )
    #             line_fit = fit_mock
    #     return line_fit

    def fit_line(self, dered_spectra, target_str: str):
        target_wave = self.rest_lambdas[target_str]
        fit_window = 25  # Half-width for the fitting window in Angstroms
        line_half_width = 5 # Half-width for estimating peak and continuum

        def make_mock_fit(continuum=0.0, std_err=0.0):
            """Creates a mock fit object for failed fits."""
            return SimpleNamespace(
                flux=0.0, err_flux=std_err, peak=0.0, cont=continuum,
                lpeak=target_wave, fwhm=1.0, err_peak=std_err,
                err_cont=std_err, err_lpeak=0.0, err_fwhm=0.0,
                fit_successful=False
            )

        try:
            # 1. Extract data in the fitting window
            sub_spec = dered_spectra.subspec(lmin=(target_wave - fit_window), lmax=(target_wave + fit_window))
            if sub_spec is None:
                return make_mock_fit()
            
            flux = np.asarray(sub_spec.data)
            if flux.size < 5:
                return make_mock_fit()

            wave = sub_spec.wave.coord()
            
            # Handle masks or NaNs
            valid = ~np.isnan(flux)
            if hasattr(sub_spec, 'mask'):
                valid &= ~sub_spec.mask
            
            wave, flux = wave[valid], flux[valid]

            if len(flux) < 5:
                return make_mock_fit()

            # 2. Initial parameter guesses
            continuum_mask = np.abs(wave - target_wave) > line_half_width
            if np.any(continuum_mask):
                cont_guess = np.ma.median(flux[continuum_mask])
                cont_std = np.ma.std(flux[continuum_mask])
            else:
                cont_guess = np.ma.median(flux)
                cont_std = np.ma.std(flux)
            
            peak_guess = np.max(flux - cont_guess)

            # 3. Define model and set parameter bounds
            g_init = models.Gaussian1D(amplitude=peak_guess, mean=target_wave, stddev=1.5)
            c_init = models.Const1D(amplitude=cont_guess)
            
            g_init.amplitude.bounds = (0, 2.0 * np.max(flux))
            g_init.mean.bounds = (target_wave - 10, target_wave + 10)
            g_init.stddev.bounds = (0.5 / 2.355, 10 / 2.355) # FWHM ~0.5-10 A
            if cont_std > 0:
                c_init.amplitude.bounds = (cont_guess - 3*cont_std, cont_guess + 3*cont_std)

            compound_model_init = g_init + c_init
            fitter = fitting.LevMarLSQFitter()
            
            # 4. Fit the model
            fit_model = fitter(compound_model_init, wave, flux, maxiter=1000)

            if fitter.fit_info['ierr'] not in [1, 2, 3, 4]:
                 raise ValueError("Fit did not converge.")

            # 5. Extract results and calculate errors
            gaussian_fit, continuum_fit = fit_model[0], fit_model[1]
            param_cov = fitter.fit_info.get('param_cov')

            if param_cov is not None:
                diag_errors = np.sqrt(np.diag(param_cov))
                err_peak, err_lpeak, err_stddev, err_cont = diag_errors
            else:
                err_peak, err_lpeak, err_stddev, err_cont = [0.0] * 4

            fwhm = gaussian_fit.stddev.value * 2.35482
            err_fwhm = err_stddev * 2.35482
            flux_val = gaussian_fit.amplitude.value * gaussian_fit.stddev.value * np.sqrt(2 * np.pi)
            
            err_flux = 0.0
            if param_cov is not None and flux_val != 0:
                amp, std = gaussian_fit.amplitude.value, gaussian_fit.stddev.value
                cov_amp_std = param_cov[0, 2]
                term1 = (err_peak / amp)**2 if amp != 0 else 0
                term2 = (err_stddev / std)**2 if std != 0 else 0
                term3 = 2 * cov_amp_std / (amp * std) if (amp * std) != 0 else 0
                err_flux_sq = (flux_val**2) * (term1 + term2 + term3)
                err_flux = np.sqrt(err_flux_sq) if err_flux_sq > 0 else 0.0

            # 6. Assemble the result object and perform sanity checks
            fit_result = SimpleNamespace(
                flux=flux_val, err_flux=err_flux, peak=gaussian_fit.amplitude.value,
                cont=continuum_fit.amplitude.value, lpeak=gaussian_fit.mean.value,
                fwhm=fwhm, err_peak=err_peak, err_cont=err_cont,
                err_lpeak=err_lpeak, err_fwhm=err_fwhm, fit_successful=True
            )
            
            if fit_result.peak + fit_result.cont > 2.0 * np.max(flux) or fit_result.flux < 0:
                 raise ValueError("Fit is unphysical.")

        except Exception:
            sub_spec_small = dered_spectra.subspec(lmin=target_wave - 10, lmax=target_wave + 10)
            if sub_spec_small is not None:
                data_small = np.asarray(sub_spec_small.data)
                if data_small.size > 2:
                    continuum = np.ma.mean(data_small)
                    std_err = np.ma.std(data_small)
                    fit_result = make_mock_fit(continuum, std_err)
                else:
                    fit_result = make_mock_fit()
            else:
                fit_result = make_mock_fit()

        return fit_result
    
    def _prepare_and_extract_spectrum(self, coords, z_guess, rad, foreground=0, cluster_member=0, lensed=0):
        id = self.makeid(coords)
        self._dirmanagement(id=id)
        
        radec = (coords.ra.deg, coords.dec.deg)
        
        # obj_spectrum, newra,newdec = self.extract_spectrum(radec[0], radec[1], radius=rad, redshift=z_guess)
        obj_spectrum = self.extract_spectrum(radec[0], radec[1], radius=rad, redshift=z_guess)
        # self.spectra[id] = obj_spectrum

        # radec = (newra,newdec)
        
        try:
            obj_z = self.find_z_from_line(obj_spectrum, z_guess)
        except:
            print(f'fit for {self.title} OIII line at {(z_guess+1)*5007} failed, using guess z')
            obj_z = z_guess
        
        rest_spectrum = self.deredshift_spectrum(obj_spectrum, obj_z)
        # self.rest_spectra[id] = rest_spectrum

        # obj_spectrum.write(f'spec/{self.title}/{id}_obs_spec.fits')
        # rest_spectrum.write(f'spec/{self.title}/{id}_rest_spec.fits')   
        
        # Prepare initial row for the table
        row_data = {
            'object_id': id,
            'ra': radec[0],
            'dec': radec[1],
            'z': obj_z,
            'foreground': foreground,
            'cluster_member': cluster_member,
            'lensed': lensed,
            'name': id
        }
        # self.ex_table.add_row(row_data)
        
        return id, radec, obj_spectrum, rest_spectrum, row_data
    
    def fit_oii_doublet(self, dered_spectra, plot=False, id=None, verbose=False):
        lmin, lmax = 3715, 3740
        sub_spec = dered_spectra.subspec(lmin=lmin, lmax=lmax)

        target_3726 = self.rest_lambdas['oii3726']
        target_3729 = self.rest_lambdas['oii3729']

        # Mock objects for failure cases
        def create_mock_fit(target_lambda):
            sub_spec_continuum = dered_spectra.subspec(lmin=target_lambda - 10, lmax=target_lambda + 10)
            continuum = np.mean(sub_spec_continuum.data) if sub_spec_continuum is not None else 0
            std_err = np.std(sub_spec_continuum.data) if sub_spec_continuum is not None else 0
            return SimpleNamespace(
                flux=0.0, err_flux=std_err, peak=0.0, cont=continuum,
                lpeak=target_lambda, fwhm=1.0, err_peak=std_err, err_cont=std_err,
                err_lpeak=0.0, err_fwhm=0.0
            )

        if sub_spec is None or sub_spec.data.ndim == 0 or sub_spec.data.shape[0] < 4:
            return create_mock_fit(target_3726), create_mock_fit(target_3729)

        wave = sub_spec.wave.coord()
        flux = sub_spec.data
        
        # Initial guesses
        cont_guess = np.ma.median(flux)
        peak_guess = np.max(flux) - cont_guess
        if peak_guess <= 0: # If no obvious peak, return mocks
             return create_mock_fit(target_3726), create_mock_fit(target_3729)

        # Astropy models
        g3726 = models.Gaussian1D(amplitude=peak_guess, mean=target_3726, stddev=1.0, bounds={'mean': (target_3726-5, target_3726+5), 'amplitude': (0, 2*peak_guess), 'stddev': (0.4, 8.5)})
        g3729 = models.Gaussian1D(amplitude=peak_guess, mean=target_3729, stddev=1.0, bounds={'mean': (target_3729-5, target_3729+5), 'amplitude': (0, 2*peak_guess)})
        continuum = models.Const1D(amplitude=cont_guess)

        # Tie standard deviations
        def tie_stddev(model):
            return model.stddev_0
        g3729.stddev.tied = tie_stddev

        doublet_model = g3726 + g3729 + continuum
        fitter = fitting.LevMarLSQFitter()

        try:
            fit = fitter(doublet_model, wave, flux)
            if fitter.fit_info.get('param_cov') is None:
                raise ValueError("Covariance matrix not computed.")
            fit_error_diag = np.sqrt(np.diag(fitter.fit_info['param_cov']))
        except Exception as e:
            # Fallback to single fits if double fit fails
            return self.fit_line(dered_spectra, 'oii3726'), self.fit_line(dered_spectra, 'oii3729')

        # Create fit objects from results
        def create_fit_ns(amplitude, mean, stddev, cont, amp_err, mean_err, stddev_err, cont_err):
            fwhm = stddev * 2.35482
            fwhm_err = stddev_err * 2.35482
            
            gauss_flux = amplitude * stddev * np.sqrt(2 * np.pi)
            if amplitude > 0 and stddev > 0:
                flux_err = gauss_flux * np.sqrt((amp_err / amplitude)**2 + (stddev_err / stddev)**2) if amplitude != 0 else 0
            else:
                flux_err = 0.0

            return SimpleNamespace(
                flux=gauss_flux, err_flux=flux_err, peak=amplitude, cont=cont,
                lpeak=mean, fwhm=fwhm, err_peak=amp_err, err_cont=cont_err,
                err_lpeak=mean_err, err_fwhm=fwhm_err
            )

        # Unpack params and errors
        # The `parameters` attribute contains all 7 model parameters, including the tied one.
        amp1, mean1, std1, amp2, mean2, _, cont = fit.parameters
        # The covariance matrix only contains entries for the 6 *free* parameters.
        err_amp1, err_mean1, err_std1, err_amp2, err_mean2, err_cont = fit_error_diag

        fit_3726 = create_fit_ns(amp1, mean1, std1, cont, err_amp1, err_mean1, err_std1, err_cont)
        fit_3729 = create_fit_ns(amp2, mean2, std1, cont, err_amp2, err_mean2, err_std1, err_cont) # Use std1 and err_std1 for tied param

        if plot and id:
            self.plot_oii_doublet(sub_spec, fit, id)

        if verbose:
            return fit_3726, fit_3729, fit
        else:
            return fit_3726, fit_3729

    def plot_oii_doublet(self, sub_spec, fit_model, id):
        wave = sub_spec.wave.coord()
        
        fig, ax = pf.create_plot(size=(4,2))
        ax.set_title(r'[OII] Doublet Fit')
        
        # Plot data
        sub_spec.plot(ax=ax, color='#ff004f', label='Data')
        
        # Plot full model
        ax.plot(wave, fit_model(wave), color='k', lw=1.4, label='Total Fit')
        
        # Plot individual components
        g1 = fit_model[0]
        g2 = fit_model[1]
        cont = fit_model[2]
        ax.plot(wave, g1(wave) + cont(wave), 'b--', label='[OII] 3726')
        ax.plot(wave, g2(wave) + cont(wave), 'g--', label='[OII] 3729')
        
        ax.set_xlabel(r'Rest Wavelength, $\lambda$, [$\AA$]')
        ax.set_ylabel(r'Flux [$\times10^{-20}\,\mathrm{erg}/\AA\,s\,\mathrm{cm}^{-2}$]')
        ax.legend(fontsize='small')

        pf.fix_plot([ax])
        fig.savefig(f'figs/{self.title}/{id}/lines/spectrum_oii_doublet.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    
    # def _fix_oii(self,rest_spectrum)
        

    def _fit_all_lines(self, rest_spectrum, id, plot=True):
        linefits = []
        linenames = list(self.rest_lambdas.keys())
        # i = 0
        # while i < len(linenames):
        #     linename = linenames[i]
            
        #     if linename == 'oii3726':
        #         oii3726_fit, oii3729_fit = self.fit_oii_doublet(rest_spectrum, plot=plot, id=id)
        #         linefits.append(oii3726_fit)
        #         linefits.append(oii3729_fit)
        #         i += 2  # Increment by 2 to skip oii3729
        #         continue
            
        #     linefit = self.fit_line(rest_spectrum, linename)
        #     linefits.append(linefit)
        #     if plot and linefit.flux != 0:
        #         self.plot_extracted_line(rest_spectrum, linefit, linename, id)
        #     i += 1
        for i,linename in enumerate(linenames):
            if linename == 'oii3726':
                oii3726_fit, oii3729_fit = self.fit_oii_doublet(rest_spectrum, plot=plot, id=id)
                linefits.append(oii3726_fit)
                linefits.append(oii3729_fit)
                continue
            elif linename == 'oii3729':
                continue
            else:
                linefit = self.fit_line(rest_spectrum, linename)
                linefits.append(linefit)
                if plot and linefit.flux != 0:
                    self.plot_extracted_line(rest_spectrum, linefit, linename, id)
            
        return linefits
    
    def _fit_balmer_lines(self, rest_spectrum,id):
        linefits = []
        for linename in self.balmer_lambda.keys():
            linefit = self.fit_line(rest_spectrum, linename)
            linefits.append(linefit)
        return linefits

    def _update_table_with_fit_results(self, pxy, linefits, rest_spectrum_id):
        locd_row = pxy
        
        # Store basic fit results
        for i, linename in enumerate(self.rest_lambdas.keys()):
            rest_wl = self.rest_lambdas.get(linename)
            linefit = linefits[i]

            # rest_spectrum = self.rest_spectra.get(rest_spectrum_id)

            # resn = rest_spectrum.wave.get_step() if rest_spectrum else 1
            resn = 0.7 

            locd_row[linename + '_flux'] = linefit.flux
            locd_row[linename + '_flux_err'] = linefit.err_flux
            locd_row[linename + '_fwhm'] = linefit.fwhm
            locd_row[linename + '_centroid'] = linefit.lpeak
            locd_row[linename + '_vel_disp'] = lr.get_velocity_disp(linefit.fwhm, rest_wl,resn)
            
            if linefit.cont != 0 and linefit.flux != 0:
                eqwidth = linefit.flux / np.abs(linefit.cont)
                locd_row[linename + '_ew'] = eqwidth
                locd_row[linename + '_ew_err'] = (linefit.err_flux / linefit.flux) * eqwidth
            else:
                locd_row[linename + '_ew'] = np.nan
                locd_row[linename + '_ew_err'] = np.nan

    def _correct_flux(self, pxy, donothing=False):
        # self.raw_table = self.ex_table.copy()
        # ebv_corr = lr.get_ebv(pxy['hbeta_flux'], pxy['hgamma_flux'])
        ebv_corr = 0.286
        flux_keys = [key for key in self.ex_table.colnames if key.endswith('_flux')]
        for key in flux_keys:
            linekey = key[:-5]
            flux = pxy[key].copy()
            if donothing:
                pxy[key] = flux
            else:
                pxy[key] = lr.correct_flux(flux,pxy[linekey+'_centroid'],ebv_corr)

    def _avg_velo_disps(self, pxy):
        global vel_keys
        vel_keys = [key for key in self.ex_table.colnames if key.endswith('_vel_disp')]
        vels = []
        for key in vel_keys:
            vels.append(pxy[key])
        vels = np.array(vels)
        # print(vels)
        pxy['mean_vel_disp'] = np.nanmean(vels)
        pxy['sterr_vel_disp'] = np.nanstd(vels)/np.sqrt(np.count_nonzero(~np.isnan(vels)))


    def _direct_mcity(self, pxy):
        metallicity, err = lr.get_metallicity_with_errors(
            pxy['oiii5007_flux'],pxy['oiii4959_flux'],pxy['oiii4363_flux'],
            pxy['oii3726_flux'],pxy['oii3729_flux'],pxy['hbeta_flux'],
            pxy['oiii5007_flux_err'],pxy['oiii4959_flux_err'],pxy['oiii4363_flux_err'],
            pxy['oii3726_flux_err'],pxy['oii3729_flux_err'],pxy['hbeta_flux_err'],
        )
        pxy['Z_dir'] = metallicity
        pxy['Z_dir_e'] = err

    def _j19_mcity(self, pxy):
        metallicity, err = lr.get_j19_with_errors(
            pxy['oiii5007_flux'],pxy['oiii4959_flux'],pxy['oii3726_flux'],
            pxy['oii3729_flux'],pxy['hbeta_flux'],
            pxy['oiii5007_flux_err'],pxy['oiii4959_flux_err'],pxy['oii3726_flux_err'],
            pxy['oii3729_flux_err'],pxy['hbeta_flux_err'],
        )
        pxy['Z_j19'] = metallicity
        pxy['Z_j19_e'] = err

    def _r23(self, pxy):
        r23, err = lr.get_R23_with_errors(
            pxy['oiii5007_flux'],pxy['oiii4959_flux'],pxy['oii3726_flux'],
            pxy['oii3729_flux'],pxy['hbeta_flux'],
            pxy['oiii5007_flux_err'],pxy['oiii4959_flux_err'],pxy['oii3726_flux_err'],
            pxy['oii3729_flux_err'],pxy['hbeta_flux_err'],
        )
        pxy['R23'] = r23
        pxy['R23_e'] = err
 
    def _update_metallicities(self, PXY):
        self._direct_mcity(PXY)
        self._j19_mcity(PXY)
        self._r23(PXY)
        return True
    
    def table_management(self, pxy, linefits, angdisp, rest_spectrum_id):
        # pxy = self.ex_table.loc[ID]
        pxy['angdisp'] = angdisp*3600 # now in arcseconds
        self._update_table_with_fit_results(pxy, linefits, rest_spectrum_id)
        
        # Create a raw version before flux correction
        raw_pxy = pxy.copy()

        self._correct_flux(pxy)
        self._update_metallicities(pxy)
        self._avg_velo_disps(pxy)
        return pxy, raw_pxy

    def pick_target(self, coords, z_guess, rad, plot=True, foreground=0, cluster_member=0, lensed=0):
        id, radec, obj_spectrum, rest_spectrum, row_data = self._prepare_and_extract_spectrum(coords, z_guess, rad, foreground, cluster_member, lensed)
        angdisp = np.sqrt((radec[0]-self.centre.ra.deg)**2+(radec[1]-self.centre.dec.deg)**2)
        
        if plot:
            self.plot_spectrum_and_cutout(obj_spectrum, radec[0], radec[1], id)
            
        linefits = self._fit_all_lines(rest_spectrum, id, plot=plot)
        balmer_linefits = self._fit_balmer_lines(rest_spectrum, id)
        
        if plot:
            self.plot_all_lines_on_spectrum(rest_spectrum, linefits, id, radec[0], radec[1])
            self.balmer_diagnostic_plot(rest_spectrum, balmer_linefits, id)
            
        # self.table_management(id,linefits, angdisp)
        final_row, raw_row = self.table_management(row_data, linefits, angdisp, id)

        return id, rest_spectrum, final_row, raw_row

    def stack_and_fit_spectra(self, plot=True):
        if not self.rest_spectra:
            print("No rest-frame spectra to stack.")
            return

        # Stack rest-frame spectra
        spectra_to_stack = list(self.rest_spectra.values())
        stacked_spectrum = spectra_to_stack[0].copy()
        for spec in spectra_to_stack[1:]:
            stacked_spectrum.data += spec.data

        id = 'STACK'
        self._dirmanagement(id=id)
        
        # Prepare table row for the stacked spectrum
        mean_z = np.mean([self.ex_table.loc[spec_id]['z'] for spec_id in self.rest_spectra.keys()])
        if id in self.ex_table['object_id']:
            self.ex_table.remove_row(np.where(self.ex_table['object_id'] == id)[0][0])

        row_data = {
            'object_id': id,
            'z': mean_z,
            'name': id
        }
        self.ex_table.add_row(row_data)

        # Fit lines and update table
        linefits = self._fit_all_lines(stacked_spectrum, id, plot=plot)
        balmer_linefits = self._fit_balmer_lines(stacked_spectrum, id)
        if plot:
            self.plot_all_lines_on_spectrum(stacked_spectrum, linefits, id)
            self.balmer_diagnostic_plot(stacked_spectrum, balmer_linefits, id)
        
        self.table_management(id,linefits, np.nan)
        return True
    
    def master_stack_and_fit_spectra(self, rest_spectra:dict, plot=True,):
        # Stack rest-frame spectra
        spectra_to_stack = list(rest_spectra.values())
        stacked_spectrum = spectra_to_stack[0].copy()
        for spec in spectra_to_stack[1:]:
            stacked_spectrum.data += spec.data

        id = 'STACK_master'
        self._dirmanagement(id=id)
        
        # stacked_spectrum.write(f'spec/{self.title}/{id}_rest_spec.fits')

        # Prepare table row for the stacked spectrum
        if id in self.ex_table['object_id']:
            self.ex_table.remove_row(np.where(self.ex_table['object_id'] == id)[0][0])

        row_data = {
            'object_id': id,
            'name': id
        }
        self.ex_table.add_row(row_data)

        # Fit lines and update table
        linefits = self._fit_all_lines(stacked_spectrum, id, plot=plot)
        balmer_linefits = self._fit_balmer_lines(stacked_spectrum, id)
        if plot:
            self.plot_all_lines_on_spectrum(stacked_spectrum, linefits, id)
            self.balmer_diagnostic_plot(stacked_spectrum, balmer_linefits, id)
        
        self.table_management(id,linefits, np.nan)
        return True

    def process_multiple_ds9(self, csv_path):
        coord_table = Table(ascii.read(csv_path))
        all_coords = SkyCoord(coord_table['ra'] * u.deg,
                              coord_table['dec'] * u.deg,
                              frame='icrs')
        for i in range(len(all_coords)):
            csv_coords = all_coords[i]
            z_estimate = coord_table['z_est'][i]
            self.pick_target(csv_coords, z_estimate, 1)
        self.stack_and_fit_spectra(plot=True)
        self.write_table()

    def _process_single_candidate(self, candidate_row):
        """Helper method to process one candidate. Designed to be called by the parallel executor."""
        csv_coords = SkyCoord(candidate_row['ra'] * u.deg, candidate_row['dec'] * u.deg, frame='icrs')
        z_estimate = (candidate_row['OIII_est'] / 5006.84) - 1
        
        description = ''
        if 'Description' in candidate_row.colnames:
            description = candidate_row['Description'].lower()

        is_foreground = 1 if 'foreground' in description else 0
        is_cluster_member = 1 if 'cluster member' in description else 0
        is_lensed = 1 if 'lensed' in description else 0

        # The pick_target method does all the work for one candidate
        id, rest_spectrum, final_row, raw_row = self.pick_target(
            csv_coords, z_estimate, 1,
            foreground=is_foreground, 
            cluster_member=is_cluster_member, 
            lensed=is_lensed
        )
        return id, rest_spectrum, final_row, raw_row

    def process_multiple_candidates(self, coord_table, zcl=np.nan, max_workers=None):
        """
        Processes multiple candidates from a table in parallel.
        
        :param coord_table: An astropy Table with candidate information.
        :param zcl: The cluster redshift.
        :param max_workers: The maximum number of processes to use. Defaults to the number of CPU cores.
        """
        all_results = []
        
        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all candidates to the executor
            future_to_candidate = {executor.submit(self._process_single_candidate, row): row for row in coord_table}
            
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    print(f'Candidate {candidate["ra"]},{candidate["dec"]} generated an exception: {exc}')

        # Unpack results and populate instance attributes
        final_rows = []
        raw_rows = []
        for id, rest_spectrum, final_row, raw_row in all_results:
            self.rest_spectra[id] = rest_spectrum
            final_rows.append(final_row)
            raw_rows.append(raw_row)

        # Create final tables from the collected rows
        if final_rows:
            self.ex_table = vstack([Table(rows=[row], names=self.column_names) for row in final_rows])
            self.raw_table = vstack([Table(rows=[row], names=self.column_names) for row in raw_rows])
        
        # Perform stacking and final table adjustments
        self.stack_and_fit_spectra(plot=True)
        self.write_table()

        self.ex_table['zcluster'] = zcl
        self.ex_table['name'] = self.ex_table['name'].astype(str)
        self.ex_table['name'] = self.title
        
        
class Candidates:
    def __init__(self, file_path: str = 'candidates.list'):
        self.file_path = file_path
        self._data = {}
        self._keys = []
        self._parse_file()
        self.analysed = False # ticker for coadd 
        

    def _parse_file(self):
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        current_key = None
        block_lines = []

        for line in lines:
            stripped_line = line.strip()
            if 'z=' in stripped_line and not stripped_line.startswith(('0', '1', '2')):
                if current_key: # Save previous block
                    self._data[current_key]['lines'] = block_lines
                    block_lines = []

                parts = stripped_line.split(' z=')
                current_key = parts[0]
                redshift = float(parts[1])
                self._keys.append(current_key)
                self._data[current_key] = {'redshift': redshift, 'lines': []}
            elif current_key and stripped_line:
                block_lines.append(stripped_line)
        
        if current_key: # Save the last block
            self._data[current_key]['lines'] = block_lines
        
        self._keys_corrected = ['/Volumes/Expansion/exp_thardy/cubes/'+(a.replace('-','m')).replace('+','p').lower()+'_COMBINED_CUBE_MED_FINAL.fits' for a in self.keys()]


    def keys(self):
        return self._keys
    
    def keys_corrected(self):
        return self._keys_corrected

    def get_candidate(self, key: str):
        if key not in self._data:
            raise ValueError(f"Key '{key}' not found in {self.file_path}")

        info = self._data[key]
        redshift = info['redshift']
        block_lines = info['lines']

        if not block_lines:
            return Table(), redshift

        # Parse data directly into lists
        ras, decs, oiii_ests, descriptions = [], [], [], []
        has_descriptions = False
        for line in block_lines:
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 3:
                continue  # Skip empty or malformed lines

            ras.append(parts[0])
            decs.append(parts[1])
            oiii_ests.append(parts[2])
            
            if len(parts) > 3:
                descriptions.append(parts[3])
                has_descriptions = True
            else:
                descriptions.append('')

        # Build the table directly
        if has_descriptions:
            table = Table({
                'RA': ras,
                'Dec': decs,
                'OIII_est': oiii_ests,
                'Description': descriptions
            })
        else:
            table = Table({
                'RA': ras,
                'Dec': decs,
                'OIII_est': oiii_ests,
            })
        if table:
            table['OIII_est'] = table['OIII_est'].astype(int)

        return table, redshift
    
    def analyse_all(self, raw=False):
        self.analysed = True

        tables = {}
        raw_tables = {}
        spectra = {}
        for i,key in enumerate(self.keys_corrected()):
            name = self.keys()[i]
            print(f'running {name}')
            tab, z = self.get_candidate(name)
            coords = SkyCoord(tab['RA'], tab['Dec'],unit=(u.hourangle, u.deg))
            tab['ra'] = coords.ra.degree
            tab['dec'] = coords.dec.degree

            hdr = Cube(key).get_wcs_header()
            indiv_cube = museCube(key,cluster_ra=hdr['CRVAL1'],cluster_dec=hdr['CRVAL2'])
            indiv_cube.process_multiple_candidates(tab, zcl=z)

            tables[name] = indiv_cube.ex_table
            raw_tables[name] = indiv_cube.raw_table
            spectra[name] = indiv_cube.rest_spectra

        self.combined_table = vstack(list(tables.values()))
        self.combined_table.write('allsources.csv', overwrite=True)

        if raw:
            self.combined_table_raw = vstack(list(raw_tables.values()))
            self.combined_table_raw.write('allsources_uncorrected.csv', overwrite=True)

        self.spectra = spectra

    def coadd(self) -> dict:
        if not self.analysed:
            self.analyse_all()
        
        # flat_spectra_dict = {
        # target+'//'+source_id: s
        # for target, sources in self.spectra.items()
        # for source_id, s in sources.items()
        # }

        example = self.spectra.get('MACS0152-28').get('28d07m28pt35s-28d53m18pt439s')

        global_min_wave = example.get_start()
        global_max_wave =example.get_end()
        step = example.get_step()

        shape = int((global_max_wave - global_min_wave) / step) + 1

        rebinned_spectra_dict = {}

        for target, sources in self.spectra.items():
            for source_id, spec in sources.items():
                new_spec = spec.resample(step=step, start=global_min_wave, shape=shape)
                rebinned_spectra_dict[target+'//'+source_id] = new_spec
        
        return rebinned_spectra_dict
        
def get_fromIFU(candidate, extras=False, locpref = '/Volumes/Expansion/exp_thardy/cubes/'):
    cluster = candidate['name']
    print(cluster)
    loc = locpref+cluster+'_COMBINED_CUBE_MED_FINAL.fits'
    with fits.open(loc) as hdul:
        cx,cy = (hdul[1].header['CRVAL1'],hdul[1].header['CRVAL2'])
        cent = SkyCoord(cx,cy, unit = u.deg)

    galloc = SkyCoord(candidate['ra'], candidate['dec'], unit=u.deg)
    cube_ift = museCube(loc, cent.ra.deg,cent.dec.deg)
    cluster = Cube(loc)
    linefits = cube_ift.pick_target(galloc,candidate['z'],1,plot=False)

    spectrum_o = cube_ift.spectra.get(list(cube_ift.spectra.keys())[0])
    rest_spectrum = cube_ift.rest_spectra.get(list(cube_ift.spectra.keys())[0])

    freq = np.linspace(rest_spectrum.get_start(), rest_spectrum.get_end(), rest_spectrum.shape[0])

    return galloc, cube_ift, cluster, spectrum_o, rest_spectrum

class QT_Candidates:
    def __init__(self, file_path: str = 'leadlines.csv'):
        self.file_path = file_path
        # self._data = {}
        self._keys = []
        self.analysed = False # ticker for coadd 
        self._initialise_file()
    
    def _initialise_file(self):
        self._leadlines = ascii.read('leadlines.csv')
        kyz = []
        for k in self._leadlines:
            kyz.append((k['dir'],k['key']))
        self._keys.append(list(set(kyz)))
        self._keys_corrected = ['/Volumes/Expansion/exp_thardy/'+d+'/'+k+'_COMBINED_CUBE_MED_FINAL.fits' for d,k in self.keys()[0]]
        self._leadlines_OIII = self._leadlines[self._leadlines['Redshift']<0.82]

    def keys(self):
        return self._keys[0]
    
    def keys_corrected(self):
        return self._keys_corrected
    
    def get_candidate(self, key: str):
        cand_leadlines = self._leadlines_OIII[self._leadlines_OIII['key'] == key]

        key = cand_leadlines['key']
        dir = cand_leadlines['dir']
        loc = '/Volumes/Expansion/exp_thardy/'+dir+'/'+key+'_COMBINED_CUBE_MED_FINAL.fits'

        # with fits.open(loc) as hdul:
        #     hdr = hdul[0].header

        with Cube(loc) as c:
            hdr = c.get_wcs_header()
        
        w = WCS(hdr)

        coords,wls = w.pixel_to_world(cand_leadlines['X_PEAK_SN'],
                                cand_leadlines['Y_PEAK_SN'],
                                cand_leadlines['Z_PEAK_SN'])
        
        cand_leadlines['ra'] = coords.ra.deg
        cand_leadlines['dec'] = coords.dec.deg

        crvals = hdr['CRVAL1'], hdr['CRVAL2']

        return cand_leadlines, cand_leadlines['zcluster'][0], crvals
        # redshift, table

    def analyse_all(self, raw=False):
        self.analysed = True

        tables = {}
        raw_tables = {}
        spectra = {}
        for i,key in enumerate(self.keys_corrected()):
            name = self.keys()[i][1]
            print(f'running {name}')
            tab, z, crvals = self.get_candidate(name)

            indiv_cube = museCube(key,cluster_ra=crvals[0],cluster_dec=crvals[1])
            indiv_cube.process_multiple_candidates(tab, zcl=z)

            tables[name] = indiv_cube.ex_table
            raw_tables[name] = indiv_cube.raw_table
            spectra[name] = indiv_cube.rest_spectra

        self.combined_table = vstack(list(tables.values()))
        self.combined_table.write('allsources.csv', overwrite=True)

        if raw:
            self.combined_table_raw = vstack(list(raw_tables.values()))
            self.combined_table_raw.write('allsources_uncorrected.csv', overwrite=True)

        self.spectra = spectra