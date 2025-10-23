import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from astropy.coordinates import SkyCoord
import sys
import plotfancy as pf
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
from astropy.table import Table
from types import SimpleNamespace
import re
from astropy.io import ascii
from astropy import units as u

import logging 
logging.getLogger('mpdaf').setLevel(logging.WARNING)

class museCube:
    def __init__(self, path:str):
        ### attributes ###
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
        if not os.path.isdir(f'figs/{self.title}/{id}'):
            if not os.path.isdir(f'figs/{self.title}'):
                if not os.path.isdir('figs'):
                    os.mkdir(f'figs')
                os.mkdir(f'figs/{self.title}')
            os.mkdir(f'figs/{self.title}/{id}')
            os.mkdir(f'figs/{self.title}/{id}/lines')
        return True

    def _get_title(self):
        pattern = r'cubes/([^_]+)_'
        match = re.search(pattern, self.path)
        if match:
            return match.group(1)
        else:
            return 'BLANK'

    def init_cube(self, PATH):
        print(f'Loading datacube: {PATH}...')
        try:
            cube = Cube(PATH)
            print(f'Successfully loaded cube. Dimensions: {cube.shape}')
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

        # --- Key Diagnostic Lines ---
        'oiii4363': 4363.21,
        'neiii':    3868.75,

        # --- Low-Ionization Lines ---
        'nii6583':  6583.45,
        'nii6548':  6548.05,
        'sii6716':  6716.44,
        'sii6731':  6730.82,

        # --- Helium Lines ---
        'heii4686': 4685.68,
        'hei5876':  5875.62,
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

        # --- Key Diagnostic Lines ---
        'oiii4363': r'[OIII] $\lambda$4363', 
        'neiii':    r'[NeIII] $\lambda$3869',

        # --- Low-Ionization Lines ---
        'nii6583':  r'[NII] $\lambda$6583',
        'nii6548':  r'[NII] $\lambda$6548',
        'sii6716':  r'[SII] $\lambda$6716',
        'sii6731':  r'[SII] $\lambda$6731',

        # --- Helium Lines (not forbidden) ---
        'heii4686': r'HeII $\lambda$4686',
        'hei5876':  r'HeI $\lambda$5876',
        }
        return True
    
    def init_table(self, cnames=None):
        self.column_names = ['object_id', 'ra', 'dec','z'] if cnames==None else cnames
        meta = ['flux','flux_err', 'ew','ew_err', 'centroid', 'fwhm']
        for key, _ in self.rest_lambdas.items():
            for m in meta:
                self.column_names.append(key+'_'+m)
        self.ex_table = Table(names=self.column_names,dtype=([str]+(len(self.column_names)-1)*[np.float64]))
        self.ex_table.add_index('object_id')
        return True
    
    ### BASIC METHODS ###
    def extract_region_centers(self,reg_file_path): #for reading ds9 reg files
        coord_pattern = re.compile(r'\w+\(([^,]+),([^,]+),.*\)')
        centers = []
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
                            centers.append((ra, dec))
                        except ValueError:
                            print(f"Warning: Could not parse coordinates from line: {line_strip}")
        except FileNotFoundError:
            print(f"Error: Region file not found at {reg_file_path}")
            return None
        return centers
    
    def write_regions_to_csv(self,reg_file_path, csv_file_path):
        coordinates = self.extract_region_centers(reg_file_path)
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

    def extract_spectrum(self, ra, dec, radius):
        center = (dec, ra)
        spec = self.cube.aperture(center, radius, is_sum=True)
        return spec
    
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

        center_pix_yx = im_cutout.wcs.sky2pix([dec, ra], 1)[0]
        aperture_circle = Circle((center_pix_yx[1], center_pix_yx[0]), 3, edgecolor='white', facecolor='none', lw=2, zorder=10)
        ax_cont.add_patch(aperture_circle)

        pf.fix_plot([ax])
        fig.savefig(f'figs/{title}/{id}/spectrum.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def plot_extracted_line(self, rest_spec, fit_params, target_str, id):
        profile = self.generate_gaussian_profile(fit_params, 500, 100)

        fig, ax = pf.create_plot(size=(4,2))

        ax.set_xlabel(r'O$_{III}$-Calibrated Rest-$\\lambda$, [$\\AA$] }')
        ax.set_ylabel(r'Flux [$\\times10^{-20}\\mathrm{erg}/\\AA\\,s\\,\\mathrm{cm}^{-2}$]')
        ax.set_title('Line Fit for'+self.lambda_keys[target_str])
        
        rest_spec.plot(ax=ax, color='#ff004f')
        ax.plot(profile[0], profile[1], color='k', lw=1.4)

        ax.set_xlim([np.min(profile[0]), np.max(profile[0])])

        pf.fix_plot([ax])
        fig.savefig(f'figs/{self.title}/{id}/lines/spectrum_{target_str}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def plot_all_lines_on_spectrum(self, rest_spec, linefits, id):
        fig, ax = pf.create_plot(size=(5, 2))

        rest_spec.plot(ax=ax, color='#ff004f', label='Spectrum')

        line_names = list(self.rest_lambdas.keys())

        for i, linefit in enumerate(linefits):
            if linefit.flux != 0:
                profile = self.generate_gaussian_profile(linefit, 500, 4)
                ax.plot(profile[0], profile[1], color='k', lw=1.2)
                
                line_name = line_names[i]
                label = self.lambda_keys[line_name]
                
                peak_y = linefit.cont + linefit.peak
                ax.text(linefit.lpeak, peak_y * 1.1, label, ha='center', va='bottom', rotation=90, fontsize=8)
    
        ax.set_xlabel(r'Rest Wavelength, $\lambda$, [$\AA$]')
        ax.set_ylabel(r'Flux [$\times10^{-20}\,\mathrm{erg}/\AA\,s\,\mathrm{cm}^{-2}$]')

        ax.set_ylim(bottom=0)
        pf.fix_plot([ax])
        fig.savefig(f'figs/{self.title}/{id}/spectrum_all_lines.png', dpi=600, bbox_inches='tight')
        plt.close(fig)


    ### EXTRACTION METHODS

    def fit_line(self,dered_spectra, target_str:str):
        target = self.rest_lambdas[target_str]
        try:
            line_fit = dered_spectra.gauss_fit(lmin=(target - 15), lmax=(target + 15), plot=False)
        except Exception:
            sub_spec = dered_spectra.subspec(lmin=target - 15, lmax=target + 15)
            
            spec_for_continuum = None
            if sub_spec is not None and len(sub_spec.shape) == 1 and sub_spec.shape[0] > 2:
                spec_for_continuum = sub_spec
            elif dered_spectra is not None and len(dered_spectra.shape) == 1 and dered_spectra.shape[0] > 2:
                spec_for_continuum = dered_spectra
            if spec_for_continuum:
                continuum = np.mean(spec_for_continuum.data)
                std_err = np.std(spec_for_continuum.data)
                
                fit_mock = SimpleNamespace(
                    flux=0.0,
                    err_flux=std_err,
                    peak=0.0,
                    cont=continuum,
                    lpeak=target,
                    fwhm=1.0,
                    err_peak=std_err,
                    err_cont=std_err,
                    err_lpeak=0.0,
                    err_fwhm=0.0
                )
                line_fit = fit_mock
        return line_fit
        
    def pick_target(self, coords, z_guess, rad, plot=True): # coords has to be an astropy skycoord obj
        obj_row = []
        id = self.makeid(coords)
        obj_row.append(id)
        self._dirmanagement(id=id)

        radec = (coords.ra.deg, coords.dec.deg)
        obj_row.extend(radec)

        obj_spectrum = self.extract_spectrum(radec[0],radec[1], radius=rad)
        self.spectra[id] = obj_spectrum

        obj_z = self.find_z_from_line(obj_spectrum,z_guess) #automatically finds oiii5007
        obj_row.append(obj_z)

        rest_spectrum = self.deredshift_spectrum(obj_spectrum, obj_z)
        self.rest_spectra[id] = rest_spectrum

        self.ex_table.add_row((obj_row+(len(self.column_names)-4)*[np.nan]))

        # plot main spectrum and continuum #
        if plot:
            self.plot_spectrum_and_cutout(obj_spectrum, radec[0], radec[1],id)

        # now deal with individual lines #
        linefits = []
        for linename, _ in self.rest_lambdas.items():
            locd_row = self.ex_table.loc[id]
            linefit = self.fit_line(rest_spectrum, linename)
            linefits.append(linefit)

            locd_row[linename+'_flux'] = linefit.flux
            locd_row[linename+'_flux_err'] = linefit.err_flux
            locd_row[linename+'_fwhm'] = linefit.fwhm
            locd_row[linename+'_centroid'] = linefit.lpeak

            eqwidth = linefit.flux/linefit.cont
            locd_row[linename+'_ew'] = eqwidth
            # locd_row[linename+'_ew_err'] = eqwidth*np.sqrt(
            #     (linefit.err_flux/linefit.flux)**2+
            #     (linefit.err_cont/linefit.cont)**2)
            locd_row[linename+'_ew_err'] = (linefit.err_flux/linefit.flux)*eqwidth if linefit.flux!=0 else np.nan

            if plot and linefit.flux!=0:
                self.plot_extracted_line(rest_spectrum, linefit, linename, id)
        
        if plot:
            self.plot_all_lines_on_spectrum(rest_spectrum, linefits, id)

    def process_multiple_ds9(self, csv_path):
        coord_table = Table(ascii.read(csv_path))
        all_coords = SkyCoord(coord_table['ra'] * u.deg,
                              coord_table['dec'] * u.deg,
                              frame='icrs')
        for i in range(len(all_coords)):
            csv_coords = all_coords[i]
            z_estimate = coord_table['z_est'][i]
            self.pick_target(csv_coords, z_estimate, 0.7)
            

        

            

        
