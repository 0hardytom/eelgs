# Create the grid without global axis sharing
fig, axes_grid = plt.subplots(4, 3,
                              figsize=(6, 8),
                              sharex=False,  # Disable global sharing
                              sharey=False,  # Disable global sharing
                              gridspec_kw={'width_ratios': [3, 3, 1],
                                           'height_ratios': [1, 3, 3, 3]})

# --- Manually link the inner scatter plot axes ---

# Link the y-axes for each row of scatter plots (rows 1, 2, 3)
for i in range(1, 4):
    axes_grid[i, 1].sharey(axes_grid[i, 0])

# Link the x-axes for each column of scatter plots (cols 0, 1)
for j in range(2):
    for i in range(2, 4):
        axes_grid[i, j].sharex(axes_grid[1, j])

# --- Hide interior tick labels to mimic 'sharex' and 'sharey' behavior ---

# Hide x-tick labels for scatter plots that are not in the bottom row
for i in range(1, 3):
    for j in range(2):
        axes_grid[i, j].tick_params(labelbottom=False)

# Hide y-tick labels for scatter plots that are not in the first column
for i in range(1, 4):
    axes_grid[i, 1].tick_params(labelleft=False)


# --- Your original plotting code ---

axes = axes_grid.flatten()

axes[2].set_visible(False)

axes[3].set_ylabel('[OIII]/[OII]')
axes[6].set_ylabel(r'([OIII]+[OII])/H$\beta$')
axes[9].set_ylabel(r'R23$-0.8\;$O32')

axes[9].set_xlabel(r'$c\Delta z / (1+z_{\mathrm{cl}})$ [$10^3$ km s$^{-1}$]')
axes[10].set_xlabel(r'$\Delta$R [arcsec]')

colours = ["#427bf6", '#ff004f']
labz = ['ELGs', 'EELGs']
metrics = [metric_all, metric]
markers = ['^', 's']

data = [O32_pop, R23_pop], [O32, R23]

for i, tabe in enumerate([elgs, selection]):
    O = data[i][0]
    R = data[i][1]
    M = R - 0.8 * O

    sns.kdeplot(x=metrics[i], ax=axes[0], color=colours[i], lw=2)
    sns.kdeplot(x=tabe['angdisp'], ax=axes[1], color=colours[i], lw=2)

    axes[3].scatter(metrics[i], np.log10(O), color=colours[i], label=labz[i], marker=markers[i])
    axes[4].scatter(tabe['angdisp'], np.log10(O), color=colours[i], label=labz[i], marker=markers[i])

    sns.kdeplot(y=np.log10(O), ax=axes[5], color=colours[i], lw=2)

    axes[6].scatter(metrics[i], np.log10(R), color=colours[i], label=labz[i], marker=markers[i])
    axes[7].scatter(tabe['angdisp'], np.log10(R), color=colours[i], label=labz[i], marker=markers[i])

    sns.kdeplot(y=np.log10(R), ax=axes[8], color=colours[i], lw=2)

    axes[9].scatter(metrics[i], np.log10(M), color=colours[i], label=labz[i], marker=markers[i])
    axes[10].scatter(tabe['angdisp'], np.log10(M), color=colours[i], label=labz[i], marker=markers[i])

    sns.kdeplot(y=np.log10(M), ax=axes[11], color=colours[i], lw=2)

plt.subplots_adjust(hspace=0.1, wspace=0.05)

axes[9].set_ylim(0, 1.3)

pf.fix_plot(axes)
