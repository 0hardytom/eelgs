


import numpy as np

# Speed of light in Angstroms per second
C_ANGSTROM_PER_S = 2.99792458e18

def jansky_to_cgs(jansky, wavelength_angstroms):
    """
    Converts a flux density from Janskys to CGS units (erg/s/cm^2/Angstrom).

    Args:
        jansky (float or np.ndarray):
            The flux density in Janskys (Jy).
        wavelength_angstroms (float or np.ndarray):
            The wavelength at which the conversion is to be made, in Angstroms (Å).

    Returns:
        float or np.ndarray:
            The flux density in CGS units (erg/s/cm^2/Å).
    """
    # 1 Jansky = 10^-23 erg/s/cm^2/Hz
    flux_cgs_hz = jansky * 1e-23

    # Convert from F_nu (per Hz) to F_lambda (per Angstrom)
    # F_lambda = F_nu * (c / lambda^2)
    flux_cgs_angstrom = flux_cgs_hz * (C_ANGSTROM_PER_S / (wavelength_angstroms**2))

    return flux_cgs_angstrom

if __name__ == '__main__':
    # Example usage:
    # Convert a flux of 1 microJansky (1e-6 Jy) at a wavelength of 5000 Å

    flux_jy = 1e-6  # 1 µJy
    wavelength_a = 5000.0  # Angstroms

    flux_cgs = jansky_to_cgs(flux_jy, wavelength_a)

    print(f"{flux_jy:.2e} Jy at {wavelength_a:.0f} Å is equivalent to {flux_cgs:.2e} erg/s/cm^2/Å")

    # Example with an array
    wavelengths = np.array([4000, 5000, 6000])
    fluxes_jy = np.array([0.5e-6, 1.0e-6, 1.5e-6])

    fluxes_cgs = jansky_to_cgs(fluxes_jy, wavelengths)

    print("\n--- Array Example ---")
    for i, w in enumerate(wavelengths):
        print(f"{fluxes_jy[i]:.2e} Jy at {w:.0f} Å is equivalent to {fluxes_cgs[i]:.2e} erg/s/cm^2/Å")
