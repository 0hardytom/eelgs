import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Assuming pandas is used for dataframes

# --- Placeholder Data ---
# NOTE: Replace this with your actual data loading for
# NONPEAS_MASSES, PEAS_MASSES, colors, and pf.
colors = ['blue', 'red']
NONPEAS_MASSES = pd.DataFrame({
    'los_velocity_kms': np.random.rand(10) * 8000,
    'angular_separation_arcsec': np.random.rand(10) * 100,
    'logmstellar': np.random.uniform(6, 11, 10),
    'logmstellar_err': np.random.rand(10) * 0.2,
    'Z_dir_gen': np.random.uniform(7.5, 10, 10),
    'Z_dir_gen_err': np.random.rand(10) * 0.1,
    'log_oiii_ew': np.random.uniform(1, 3, 10),
    'oiii5007_ew': np.random.uniform(10, 100, 10),
    'oiii5007_ew_err': np.random.rand(10) * 5,
    'dMS': np.random.uniform(-1, 1, 10),
})
PEAS_MASSES = pd.DataFrame({
    'los_velocity_kms': np.random.rand(10) * 8000,
    'angular_separation_arcsec': np.random.rand(10) * 100,
    'logmstellar': np.random.uniform(6, 11, 10),
    'logmstellar_error': np.random.rand(10) * 0.2, # Note the different error column name
    'Z_dir_gen': np.random.uniform(7.5, 10, 10),
    'Z_dir_gen_err': np.random.rand(10) * 0.1,
    'log_oiii_ew': np.random.uniform(1, 3, 10),
    'oiii5007_ew': np.random.uniform(10, 100, 10),
    'oiii5007_ew_err': np.random.rand(10) * 5,
    'dMS': np.random.uniform(-1, 1, 10),
})
class PlotFixer:
    def fix_plot(self, axes):
        pass # Dummy class
pf = PlotFixer()
# --- End Placeholder Data ---


fig, axs = plt.subplots(ncols=2, nrows=4,figsize=(12, 5.33), sharex='col', sharey='row')
fig.subplots_adjust(wspace=0.1/3, hspace=0.1)
axes = axs.flatten()

# Hide x-tick labels for the top three rows
for i in range(6):
    plt.setp(axes[i].get_xticklabels(), visible=False)

# Hide y-tick labels for the right-hand column
for i in [1, 3, 5, 7]:
    plt.setp(axes[i].get_yticklabels(), visible=False)


axes[0].set_ylabel(r'log $M_{*}$')
axes[2].set_ylabel(r'Z [Z$_{\odot}$]')
axes[4].set_ylabel(r'log EW[OIII]')
axes[6].set_ylabel(r'dMS') # Y-label for the new row

# Move x-labels to the new bottom row
axes[6].set_xlabel(r'Velocity$_{LoS, \mathrm{\to BCG}}$ [$10^3$ km s$^{-1}$]')
axes[7].set_xlabel(r'BCG $\Delta$R [arcsecond]')

keys = ['logmstellar','Z_dir_gen','log_oiii_ew', 'dMS'] # Added 'dMS'
keys2 = ['los_velocity_kms','angular_separation_arcsec']

# Add dummy error for dMS as it is negligible
NONPEAS_MASSES['dMS_err'] = 0
PEAS_MASSES['dMS_err'] = 0

NONPEAS_MASSES['log_oiii_ew_err'] = 0.432*NONPEAS_MASSES['oiii5007_ew_err']/NONPEAS_MASSES['oiii5007_ew']
PEAS_MASSES['log_oiii_ew_err'] = 0.432*PEAS_MASSES['oiii5007_ew_err']/PEAS_MASSES['oiii5007_ew']

NONPEAS_MASSES['logmstellar_err'] = NONPEAS_MASSES['logmstellar_err']
PEAS_MASSES['logmstellar_err'] = PEAS_MASSES['logmstellar_error']

for i,a in enumerate(axs):
    for ii,ax in enumerate(a):
        for iii, tab in enumerate([NONPEAS_MASSES,PEAS_MASSES]):
            normaliser = 10**(3-3*ii)# makes kms in 10^3
            ax.errorbar(tab[keys2[ii]]/normaliser, tab[keys[i]], yerr=2*tab[keys[i]+'_err'], fmt='s', capsize=2, ecolor='gray', ms=4, color=colors[iii])

axes[6].set_xlim(0,8) # Set x-limit on the new bottom row
axes[0].set_ylim(6.1,11.1)
axes[2].set_ylim(7.5,10.1)
# You may want to add a ylim for the new dMS plot, e.g.:
# axes[6].set_ylim(-1.5, 1.5)

pf.fix_plot(axes)
plt.savefig('figs/cluster_structure_modified.png', bbox_inches='tight', dpi=600,
            facecolor='white')

print("Modified script saved to modified.py and plot saved to figs/cluster_structure_modified.png")
