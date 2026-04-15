import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnchoredText
import os

SOLAR_Z = 8.69  # 12 + log(O/H) solar
COLS   = ['logSFR', 'Z_dir_gen', 'log_oiii_ew']
LABELS = [
    r'$\log(\mathrm{SFR_{Kenn.}}$'+'\n'+r'$[M_\odot\ yr^{-1}])$',
    r'$12+\log(\mathrm{O/H})$',
    r'$\log_{10}(EW\mathrm{[OIII]_{5007}})$',
]
N      = len(COLS)
LEVELS = 5
GAP    = 1   # number of empty columns between the two corner plots

STYLES = {
    'KS': {'non': '#41afea', 'pea': '#ff004f'},
    'MW': {'non': '#175475ff', 'pea': '#8e012dff'},
}

# This is placeholder data. Replace with your actual data loading.
# Creating dummy data for demonstration purposes
def make_dummy_data(n_samples, seed):
    np.random.seed(seed)
    data = {
        'logSFR': np.random.normal(0, 1, n_samples),
        'Z_dir_gen': np.random.normal(8.5, 0.2, n_samples),
        'oiii5007_ew': 10**np.random.normal(2, 0.5, n_samples),
        'hbeta_ew': 10**np.random.normal(1, 0.3, n_samples),
        'WFoiii5007_ew': 10**np.random.normal(2.2, 0.6, n_samples),
        'WFhbeta_ew': 10**np.random.normal(1.2, 0.4, n_samples),
    }
    return pd.DataFrame(data)

MASTER_CLEAN = make_dummy_data(200, 42)
MUSEWIDE_CLEAN = make_dummy_data(150, 123)


# ── Precompute log EW ─────────────────────────────────────────────────────────
PEAS_MASK    = (np.log10(MASTER_CLEAN['oiii5007_ew']) > 2) | \
               (np.log10(MASTER_CLEAN['hbeta_ew']) > 1)
PEAS         = MASTER_CLEAN[PEAS_MASK].copy()
NONPEAS      = MASTER_CLEAN[~PEAS_MASK].copy()
PEAS['log_oiii_ew']    = np.log10(PEAS['oiii5007_ew'])
NONPEAS['log_oiii_ew'] = np.log10(NONPEAS['oiii5007_ew'])

MW_PEAS_MASK = (np.log10(MUSEWIDE_CLEAN['WFoiii5007_ew']) > 2) | \
               (np.log10(MUSEWIDE_CLEAN['WFhbeta_ew']) > 1)
MW_PEAS      = MUSEWIDE_CLEAN[MW_PEAS_MASK].copy()
MW_NONPEAS   = MUSEWIDE_CLEAN[~MW_PEAS_MASK].copy()
MW_PEAS['log_oiii_ew']    = np.log10(MW_PEAS['WFoiii5007_ew'])
MW_NONPEAS['log_oiii_ew'] = np.log10(MW_NONPEAS['WFoiii5007_ew'])

# ── Single flat GridSpec: N + GAP + N columns, N rows ─────────────────────────
fig = plt.figure(figsize=(14, 6.5))

total_cols = N + GAP + N
gs = gridspec.GridSpec(
    N, total_cols,width_ratios=[1]*N + [0.3] + [1]*N,  # shrink the gap column
    hspace=0.06, wspace=0.03,
    left=0.07, right=0.97, top=0.88, bottom=0.10, # Adjusted top for space
)

# Build axes arrays by indexing gs directly
def make_axes_from_gs(col_offset):
    axes = [[None]*N for _ in range(N)]
    legend_ax = None
    for row in range(N):
        for col in range(N):
            if col <= row:
                axes[row][col] = fig.add_subplot(gs[row, col_offset + col])
            elif row == 0 and col == N-1:
                # Top-right cell — use for label + legend
                legend_ax = fig.add_subplot(gs[0:2, col_offset + 1: col_offset + N])
                legend_ax.set_axis_off()
    return axes, legend_ax

axes_ks, leg_ax_ks = make_axes_from_gs(0)
axes_mw, leg_ax_mw = make_axes_from_gs(N + GAP)

# ── Sharing: do it after all axes exist ───────────────────────────────────────
def apply_sharing(axes):
    # x: share each column's cells with the bottom cell of that column
    for col in range(N):
        ref = axes[N-1][col]
        for row in range(col, N-1):
            axes[row][col].sharex(ref)
    # y: share each row's off-diagonal cells with the leftmost (col=0)
    for row in range(1, N):
        ref = axes[row][0]
        for col in range(1, row):
            axes[row][col].sharey(ref)

apply_sharing(axes_ks)
apply_sharing(axes_mw)

# ── Fill ──────────────────────────────────────────────────────────────────────
def fill_corner(axes, nonpeas_df, peas_df, style_key, survey_label):
    c_non = STYLES[style_key]['non']
    c_pea = STYLES[style_key]['pea']

    for row in range(N):
        for col in range(N):
            ax = axes[row][col]
            if ax is None:
                continue

            xcol = COLS[col]
            ycol = COLS[row]

            if row == col:
                # Plot KDEs for both populations
                sns.kdeplot(nonpeas_df[xcol], ax=ax, color=c_non, fill=True, alpha=0.3)
                sns.kdeplot(nonpeas_df[xcol], ax=ax, color=c_non, lw=2.5)
                sns.kdeplot(peas_df[xcol],    ax=ax, color=c_pea,fill=True, alpha=0.3)
                sns.kdeplot(peas_df[xcol],    ax=ax, color=c_pea, lw=2.5)
                
                # Calculate and display stats for nonpeas
                data_non = nonpeas_df[xcol].values
                cleaned_data_non = data_non[~np.isnan(data_non)]
                if cleaned_data_non.size > 0:
                    q1_non, med_non, q3_non = np.percentile(cleaned_data_non, [25, 50, 75])
                    upper_err_non = q3_non - med_non
                    lower_err_non = med_non - q1_non
                    stat_text_non = fr'${med_non:.2f}^{{+{upper_err_non:.2f}}}_{{-{lower_err_non:.2f}}}$'
                    ax.text(0.05, 1.05, stat_text_non,
                            ha='left', va='bottom', transform=ax.transAxes,
                            color=c_non, fontsize=12, fontweight='bold', clip_on=False)

                # Calculate and display stats for peas
                data_pea = peas_df[xcol].values
                cleaned_data_pea = data_pea[~np.isnan(data_pea)]
                if cleaned_data_pea.size > 0:
                    q1_pea, med_pea, q3_pea = np.percentile(cleaned_data_pea, [25, 50, 75])
                    upper_err_pea = q3_pea - med_pea
                    lower_err_pea = med_pea - q1_pea
                    stat_text_pea = fr'${med_pea:.2f}^{{+{upper_err_pea:.2f}}}_{{-{lower_err_pea:.2f}}}$'
                    ax.text(0.95, 1.05, stat_text_pea,
                            ha='right', va='bottom', transform=ax.transAxes,
                            color=c_pea, fontsize=12, fontweight='bold', clip_on=False)

                ax.set_ylim(bottom=0)
                ax.yaxis.set_visible(False)

            else:
                sns.kdeplot(x=nonpeas_df[xcol], y=nonpeas_df[ycol], ax=ax,
                            color=c_non, fill=True, alpha=0.4, levels=LEVELS, zorder=0)
                sns.kdeplot(x=nonpeas_df[xcol], y=nonpeas_df[ycol], ax=ax,
                            color='k', lw=0.8, fill=False, alpha=0.15, levels=LEVELS, zorder=1)
                sns.kdeplot(x=peas_df[xcol], y=peas_df[ycol], ax=ax,
                            color=c_pea, fill=True, alpha=0.45, levels=LEVELS, zorder=2)
                sns.kdeplot(x=peas_df[xcol], y=peas_df[ycol], ax=ax,
                            color='k', lw=0.8, fill=False, alpha=0.7, levels=LEVELS, zorder=3)
                
                if ycol == 'Z_dir_gen' and row != col:
                    ax.axhline(SOLAR_Z, color='gray', lw=2.5, alpha=0.5, zorder=0)
                if xcol == 'Z_dir_gen' and row == col:
                    ax.axvline(SOLAR_Z, color='gray', lw=2.5, alpha=0.5, zorder=0)

            # x labels/ticks: bottom row only
            if row == N - 1:
                ax.set_xlabel(LABELS[col], fontsize=16)
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False,labelsize=18)

            # y labels/ticks: leftmost column, off-diagonal only
            if col == 0 and row != 0:
                ax.set_ylabel(LABELS[row], fontsize=16)
                ax.tick_params(axis='y', labelleft=True,labelsize=18)
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False,labelsize=18)

    # Survey label in top-left cell
    at = AnchoredText(survey_label, loc='upper right',
                      prop=dict(size=12, fontweight='bold'),
                      frameon=True, pad=0.4)
    at.patch.set_boxstyle("round,pad=0.1")
    at.patch.set_edgecolor('black')
    at.patch.set_linewidth(1.5)
    at.patch.set_facecolor('white')
    at.patch.set_alpha(0.8)
    # axes[0][0].add_artist(at)

fill_corner(axes_ks, NONPEAS,    PEAS,    'KS', 'Kaleidoscopes')
fill_corner(axes_mw, MW_NONPEAS, MW_PEAS, 'MW', 'MUSE-WIDE')

for axes in [axes_ks, axes_mw]:
    col = COLS.index('Z_dir_gen')
    for row in range(col, N):
        axes[row][col].axvline(SOLAR_Z, color='gray', lw=2.5, alpha=0.5, zorder=0)

for col in range(N):
    for row in range(col, N):
        xlim = axes_ks[row][col].get_xlim()
        axes_mw[row][col].set_xlim(xlim)
        if row != col:
            ylim = axes_ks[row][col].get_ylim()
            axes_mw[row][col].set_ylim(ylim)

# ── Legend ────────────────────────────────────────────────────────────────────
for leg_ax, label, style_key in [
    (leg_ax_ks, 'Kaleidoscopes', 'KS'),
    (leg_ax_mw, 'MUSE-WIDE',     'MW'),
]:
    pop_handles = [
        Patch(facecolor=STYLES[style_key]['non'], alpha=0.6, label='sELGs'),
        Patch(facecolor=STYLES[style_key]['pea'], alpha=0.6, label='EELGs'),
    ]
    pos = leg_ax.get_position()
    mid_x = (pos.x0 + pos.x1) / 1.83
    mid_y = (pos.y0 + pos.y1) / 1.63

    leg = fig.legend(handles=pop_handles,
                     title=label,
                     loc='center',
                     bbox_to_anchor=(mid_x, mid_y),
                     bbox_transform=fig.transFigure,
                     ncol=1, fontsize=20,
                     title_fontsize=25,
                     frameon=True, edgecolor='black',
                     facecolor='white', framealpha=0.9)
    leg.get_title().set_fontweight('bold')

def fix_plot(axes):
    for ax in axes:
        ax.tick_params(direction='in', top=True, right=True)

all_axes = [ax for row in axes_ks + axes_mw for ax in row if ax is not None]
fix_plot(all_axes)


for axes in [axes_ks, axes_mw]:
    for col in range(N):
        axes[N-1][col].tick_params(axis='x', labelsize=18)
    for row in range(1, N):
        axes[row][0].tick_params(axis='y', labelsize=18)

if not os.path.exists('figs'):
    os.makedirs('figs')

plt.savefig('figs/metals_sfr_corner.pdf',
            bbox_inches='tight', dpi=500,
            facecolor='white', format='pdf')

print("Script finished. Plot saved to figs/metals_sfr_corner.pdf")