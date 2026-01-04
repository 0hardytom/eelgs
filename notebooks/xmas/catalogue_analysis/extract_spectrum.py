import numpy as np
from mpdaf.obj import Cube

class IFUExtraction:
    def __init__(self, cube_path):
        """
        Initializes the IFU extraction class.

        Parameters
        ----------
        cube_path : str
            Path to the FITS file of the MPDAF cube.
        """
        self.cube = Cube(cube_path)

    def extract_spectrum(self, ra, dec, radius, redshift):
        """
        Extracts a spectrum from the cube by first determining the precise center
        of the object based on the [OIII] emission line.

        This method creates a narrow-band image around the observed wavelength of
        [OIII]5007, finds the flux peak in that image, and uses that peak's
        location as the center for a circular aperture extraction on the full cube.

        Parameters
        ----------
        ra : float
            Initial Right Ascension of the target's center.
        dec : float
            Initial Declination of the target's center.
        radius : float
            Radius of the circular aperture for spectrum extraction in arcseconds.
        redshift : float
            The redshift of the target, used to locate the [OIII] line.

        Returns
        -------
        spec : mpdaf.obj.Spectrum
            The extracted spectrum from the corrected center.
        corrected_ra : float
            The RA coordinate of the [OIII] emission peak.
        corrected_dec : float
            The Dec coordinate of the [OIII] emission peak.
        """
        # Define the rest-frame wavelength of [OIII]5007 in Angstroms
        OIII_REST_WAVE = 5007.0

        # Calculate the observed wavelength of the [OIII] line using the redshift
        oiii_observed_wave = OIII_REST_WAVE * (1 + redshift)

        # Define a small wavelength window (e.g., 20 Angstroms) around the line
        # to create a sub-cube. This isolates the line emission.
        wave_window = 20.0
        lambda_min = oiii_observed_wave - (wave_window / 2)
        lambda_max = oiii_observed_wave + (wave_window / 2)

        # Create a sub-cube centered on the [OIII] line
        subcube = self.cube.select_lambda(lambda_min, lambda_max)

        # Create a 2D image from the sub-cube by summing the flux along the wavelength axis
        oiii_image = subcube.sum(axis=0)

        # Find the coordinates of the peak flux in the [OIII] image.
        # This gives us the most accurate center of the object's emission.
        peak_info = oiii_image.peak()
        corrected_dec = peak_info['dec']
        corrected_ra = peak_info['ra']

        # Define the new, corrected center for the aperture extraction
        corrected_centre = (corrected_dec, corrected_ra)

        # Extract the spectrum from the *original* full cube using the corrected center
        spec = self.cube.aperture(corrected_centre, radius, is_sum=True)

        # Return the final spectrum and the corrected coordinates
        return spec, corrected_ra, corrected_dec
