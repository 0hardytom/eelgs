import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from prospect.io import read_results as pread

# This function is the same as in your run script. It's needed to convert
# the z_fraction posteriors into mass fractions.
def zfrac_to_masses_log(logmass=None, z_fraction=None, agebins=None, **extras):
    """Converts z_fraction posteriors to mass fractions."""
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])
    
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    if (mass_fraction < 0).any():
        idx = mass_fraction < 0
        if np.isclose(mass_fraction[idx], 0, rtol=1e-8):
            mass_fraction[idx] = 0.0
        else:
            raise ValueError('The input z_fractions are returning negative masses!')

    masses = 10**logmass * mass_fraction
    return masses

def get_model_params(results):
    """
    Extracts model parameters from the results dictionary.
    Tries to get them from the 'model' object, falls back to manual reconstruction.
    """
    try:
        # This is the robust way, if the model object was saved
        model = results['model']
        theta_labels = model.theta_labels()
        agebins = model.params['agebins'].T
        zred = model.params['zred']
    except KeyError:
        # Fallback for older files that might be missing the model blob
        print("Warning: 'model' object not found in HDF5 file. Reconstructing parameters manually.")
        print("This may be inaccurate if you have changed the model since the fit was run.")
        nparams = results['chain'].shape[1]
        nbins = 10
        z_fraction_labels = [f"z_fraction_{i}" for i in range(nbins - 1)]
        theta_labels = (['dust2', 'duste_gamma', 'duste_umin', 'duste_qpah', 
                         'logmass', 'logzsol'] + 
                        z_fraction_labels + 
                        ['gas_logz', 'gas_logu'])

        tuniv = 14.
        tbinmax = (tuniv * 0.8) * 1e9
        lim1, lim2 = 7.0, 8.0
        agelims = [0, lim1] + np.linspace(lim2, np.log10(tbinmax), nbins - 2).tolist() + [np.log10(tuniv * 1e9)]
        agebins = np.array([agelims[:-1], agelims[1:]])
        zred = 0.20275 # Hardcoded for the first galaxy in your sheet

        if len(theta_labels) != nparams:
            raise ValueError(f"Parameter mismatch! Chain has {nparams} params, but manual reconstruction has {len(theta_labels)}.")

    return theta_labels, agebins, zred


def plot_sfh(results_file):
    """
    Loads Prospector results and plots the star formation history.

    Args:
        results_file (str): Path to the HDF5 results file.
    """
    # --- Load Data using the official Prospector reader ---
    try:
        results, _, _ = pread.results_from(results_file)
    except Exception as e:
        print(f"Error loading file with prospect.io.read_results: {e}")
        return

    flatchain = results['chain']
    if flatchain.size == 0:
        print(f"Error: The chain in {results_file} is empty even when loaded with pread.")
        return

    # --- Get Model Info ---
    try:
        theta_labels, agebins, zred = get_model_params(results)
    except (ValueError, KeyError) as e:
        print(f"Error getting model parameters: {e}")
        return

    # --- Process Posteriors ---
    logmass_idx = theta_labels.index('logmass')
    z_fraction_indices = [i for i, label in enumerate(theta_labels) if 'z_fraction' in label]

    # --- Calculate SFR for each posterior sample ---
    all_sfrs = []
    for sample in flatchain:
        logmass_sample = sample[logmass_idx]
        z_fraction_sample = sample[z_fraction_indices]
        
        masses = zfrac_to_masses_log(logmass=logmass_sample, z_fraction=z_fraction_sample, agebins=agebins)
        bin_durations = 10**agebins[1] - 10**agebins[0]
        sfr = masses / bin_durations
        all_sfrs.append(sfr)

    all_sfrs = np.array(all_sfrs)

    # --- Convert to Cosmic Time ---
    lookback_time_bins = 10**agebins / 1e9
    t_universe_at_z = cosmo.age(zred).value
    cosmic_time_bins = t_universe_at_z - lookback_time_bins

    # --- Calculate Percentiles ---
    sfr_percentiles = np.percentile(all_sfrs, [16, 50, 84], axis=0)
    sfr_16, sfr_50, sfr_84 = sfr_percentiles

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_time = np.insert(cosmic_time_bins[1, :], 0, cosmic_time_bins[0, 0])
    plot_sfr_50 = np.insert(sfr_50, 0, sfr_50[0])
    
    ax.step(plot_time, plot_sfr_50, where='pre', color='blue', lw=2, label='Median SFH')
    
    ax.fill_between(plot_time, np.insert(sfr_16, 0, sfr_16[0]), np.insert(sfr_84, 0, sfr_84[0]), 
                    step='pre', color='blue', alpha=0.2, label='16th-84th Percentile')

    ax.set_xlabel('Cosmic Time (Gyr)')
    ax.set_ylabel(r'Star Formation Rate ($M_\odot / yr$)')
    ax.set_title(f'Star Formation History for Galaxy at z={zred:.2f}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.set_xlim(0, t_universe_at_z)
    if np.any(sfr_16 > 0):
        ax.set_ylim(bottom=np.min(sfr_16[sfr_16 > 0]) / 2, top=np.max(sfr_84) * 2)

    plt.savefig('sfh_plot.png', dpi=300)
    print("Plot saved to sfh_plot.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Star Formation History from Prospector results.")
    parser.add_argument("results_file", type=str, help="Path to the HDF5 results file (e.g., test000_fit.h5).")
    args = parser.parse_args()
    
    plot_sfh(args.results_file)