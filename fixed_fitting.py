# fixed_fitting.py

import numpy as np
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyWarning
import warnings
from types import SimpleNamespace

# Suppress annoying warnings from astropy fitter, which can be verbose
warnings.simplefilter('ignore', category=AstropyWarning)

# The original function was a method of a class.
# Below is the rewritten method. You should integrate it into your existing class.
# A placeholder class `SpectrumFitter` is used here for demonstration.

class SpectrumFitter:
    """
    A placeholder class to contain the fit_line method.
    The user should integrate this method into their existing class structure.
    """
    def __init__(self, rest_lambdas):
        """
        Initialize with a dictionary of rest-frame wavelengths for emission lines.
        Example: {'OII3727': 3727.09, 'OIII5007': 5007.0}
        """
        self.rest_lambdas = rest_lambdas

    def fit_line(self, dered_spectra, target_str: str):
        """
        Fits a Gaussian profile to a spectral line using astropy.modeling.

        This function is designed to be more robust than the original implementation,
        addressing issues with unphysical fits by using astropy.modeling with
        parameter bounds and better initial guesses. It fits a Gaussian profile
        plus a constant continuum level.

        Parameters
        ----------
        dered_spectra : mpdaf.obj.Spectrum
            The 1D spectrum to fit. It is assumed to be an object with methods
            like .subspec() and attributes like .wave.coord() and .data,
            consistent with an mpdaf.obj.Spectrum.
        target_str : str
            The name of the line to fit (e.g., 'OII3727'). Must be a key in
            self.rest_lambdas.

        Returns
        -------
        types.SimpleNamespace
            An object containing the fit results, mimicking the output of
            mpdaf's gauss_fit. Includes a 'fit_successful' boolean flag.
            If the fit fails, it returns a mock object with zero flux.
        """
        target_wave = self.rest_lambdas[target_str]
        fit_window = 50  # Half-width for the fitting window in Angstroms
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

# Example usage:
if __name__ == '__main__':
    # This is a mock of an mpdaf.obj.Spectrum for demonstration purposes.
    # In your code, you would pass your actual mpdaf Spectrum object.
    class MockWave:
        def coord(self):
            return np.linspace(4800, 5200, 1000)

    class MockSpectrum:
        def __init__(self, wave, data, mask=None):
            self.wave = wave
            self.data = data
            self.mask = mask if mask is not None else np.zeros_like(data, dtype=bool)

        def subspec(self, lmin, lmax):
            indices = (self.wave.coord() >= lmin) & (self.wave.coord() <= lmax)
            return MockSpectrum(MockWave(), self.data[indices], self.mask[indices])

    # Create a fake spectrum with a Gaussian line and noise
    wave_coords = np.linspace(4800, 5200, 1000)
    true_continuum = 1.0
    true_amplitude = 5.0
    true_mean = 5007.0
    true_stddev = 2.0
    
    g = models.Gaussian1D(amplitude=true_amplitude, mean=true_mean, stddev=true_stddev)
    c = models.Const1D(true_continuum)
    
    flux_data = g(wave_coords) + c(wave_coords)
    noise = np.random.normal(0., 0.2, flux_data.shape)
    flux_data += noise

    mock_spectrum = MockSpectrum(MockWave(), flux_data)

    # --- How to use the fitter ---
    
    # 1. Define your rest-frame wavelengths
    rest_lambdas = {'OIII5007': 5007.0}

    # 2. Instantiate the fitter class
    fitter = SpectrumFitter(rest_lambdas)

    # 3. Call the fit_line method
    fit_result = fitter.fit_line(mock_spectrum, 'OIII5007')

    # 4. Print the results
    if fit_result.fit_successful:
        print("Fit successful!")
        print(f"  Flux: {fit_result.flux:.2f} +/- {fit_result.err_flux:.2f}")
        print(f"  Peak: {fit_result.peak:.2f} +/- {fit_result.err_peak:.2f}")
        print(f"  Continuum: {fit_result.cont:.2f} +/- {fit_result.err_cont:.2f}")
        print(f"  Wavelength: {fit_result.lpeak:.2f} +/- {fit_result.err_lpeak:.2f}")
        print(f"  FWHM: {fit_result.fwhm:.2f} +/- {fit_result.err_fwhm:.2f}")
    else:
        print("Fit failed.")
        print(f"  Estimated Continuum: {fit_result.cont:.2f}")
