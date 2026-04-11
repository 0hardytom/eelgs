import numpy as np
import pandas as pd
import corner
from matplotlib.ticker import MaxNLocator

# Assuming 'corrected_stellar_mass_posterior', 'results', 'idx', 'model_params', 'model_sfr' are defined elsewhere

data = {'log(M$_*$/M$_{\odot}$)\n': np.log10(corrected_stellar_mass_posterior),
       'log(Z$_*$/Z$_{\odot}$)\n': results['chain'][idx,model_params.index('logzsol')],
       'log(SFR)\n': np.log10(model_sfr),
       'A$_V$\n': results['chain'][idx,model_params.index('dust2')],
       'B$_V$\n': np.log10(results['chain'][idx,model_params.index('duste_gamma')]),
        'log$\;U_{\mathrm{gas}}$\n': results['chain'][idx,model_params.index('gas_logu')],
}

df = pd.DataFrame(data)

# Define priors for each parameter.
priors = [None, None, None, None, None, None]

# Define the significant figures for each parameter's title.
sig_figs = [5, 5, 3, 4, 2, 5]
# Create a list of format strings for the titles, which is the correct way to do this in `corner`.
title_fmts = [f'.{sf}g' for sf in sig_figs]


figure = corner.corner(df, labels=list(data.keys()),
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,
                       title_fmt=title_fmts,  # Use the built-in formatter
                       title_kwargs={"fontsize": 25},
                       label_kwargs={'fontsize': 25},
                       labelpad=0.15,
                       bins=25,
                       plot_datapoints=False,
                       priors=priors)

# Set number of ticks to 3 for all visible axes that `corner` has created.
# This is a safer way to modify the plot without breaking the layout.
for ax in figure.get_axes():
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
