import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import corner
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from decimal import Decimal

FONTSIZE = 15

plt.rcParams.update({
    'font.size':        FONTSIZE,
    'axes.titlesize':   FONTSIZE,
    'axes.labelsize':   FONTSIZE,
    'xtick.labelsize':  FONTSIZE,
    'ytick.labelsize':  FONTSIZE,
    'legend.fontsize':  FONTSIZE,
})

# ---------------------------------------------------------------------------
# Utility: round x to the decimal place of the first significant figure of ref
# ---------------------------------------------------------------------------
def round_to_1sf_of(ref):
    """Return the number of decimal places needed to express ref to 1 s.f."""
    if ref == 0:
        return 0
    from math import floor, log10
    mag = floor(log10(abs(ref)))   # e.g. 0.0034 → mag = -3
    return max(0, -mag)            # decimal places needed

def format_title(val, low, high):
    """
    Round each error to 1 s.f., then round val to match the precision of
    whichever rounded error is more precise (i.e. more decimal places).
    """
    def round_err_to_1sf(e):
        if e == 0:
            return 0.0, 0
        from math import floor, log10
        mag   = floor(log10(abs(e)))
        scale = 10 ** mag
        rounded = round(e / scale) * scale   # 1 s.f.
        dp = max(0, -mag)
        return rounded, dp

    low_r,  dp_low  = round_err_to_1sf(low)
    high_r, dp_high = round_err_to_1sf(high)
    dp = max(dp_low, dp_high)          # use the finer of the two precisions

    val_r = round(val, dp)

    fmt = f'{{:.{dp}f}}'
    return f'${fmt.format(val_r)}^{{+{fmt.format(high_r)}}}_{{-{fmt.format(low_r)}}}$'


def add_break_markers(ax_left, ax_right, size=6, angle=70):
    import matplotlib.transforms as mtransforms

    dx = size * np.cos(np.radians(angle))
    dy = size * np.sin(np.radians(angle))

    for ax, x_ax in [(ax_left, 1.0), (ax_right, 0.0)]:
        for y_frac in [0.0, 1.0]:
            disp_pt = ax.transAxes.transform((x_ax, y_frac))
            x0_disp, y0_disp = disp_pt
            p1 = ax.transAxes.inverted().transform((x0_disp - dx, y0_disp - dy))
            p2 = ax.transAxes.inverted().transform((x0_disp + dx, y0_disp + dy))
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    transform=ax.transAxes,
                    color='k', linewidth=1, clip_on=False)


param_keys = ['logmass', 'logzsol', None, 'dust2', 'duste_gamma', 'gas_logu']
labels     = list(data.keys())
n_params   = len(data)
df         = pd.DataFrame(data)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig     = plt.figure(figsize=(8.27, 11.69))
subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 4], hspace=0.09)

ax_sfh       = subfigs[0].add_subplot(1, 1, 1)
spec_subfigs = subfigs[1].subfigures(1, 2, width_ratios=[3, 1], wspace=0.0)
ax_spec_opt  = spec_subfigs[0].add_subplot(1, 1, 1)
ax_spec_ir   = spec_subfigs[1].add_subplot(1, 1, 1)

ax_spec_ir.sharey(ax_spec_opt)
ax_spec_ir.tick_params(labelleft=False)

# ---------------------------------------------------------------------------
# SFH panel
# ---------------------------------------------------------------------------
agebins      = model.params['agebins']
time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
zred         = float(model.params['zred'])
tuniv_gyr    = cosmo.age(zred).value

n_draw      = min(200, len(idx))
sfh_samples = []

for i in np.random.choice(idx, size=n_draw, replace=False):
    logmass        = results['chain'][i, model_params_names.index('logmass')]
    sfr_ratio_idxs = [k for k, l in enumerate(model_params_names) if l.startswith('logsfr_ratios')]
    logsfr_ratios  = results['chain'][i, sfr_ratio_idxs]
    masses         = logsfr_ratios_to_masses(
        logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins,
    )
    sfh_samples.append(masses / time_per_bin)

sfh_samples = np.array(sfh_samples)
sfh_lo      = np.percentile(sfh_samples, 16, axis=0)
sfh_med     = np.percentile(sfh_samples, 50, axis=0)
sfh_hi      = np.percentile(sfh_samples, 84, axis=0)

plot_sfh_step_continuous(ax_sfh, agebins, sfh_lo, sfh_hi, sfh_med)

ax_sfh.set_xlim(0, tuniv_gyr)
ax_sfh.set_yscale('log')
ax_sfh.set_xlabel('Lookback time [Gyr]')
ax_sfh.set_ylabel('SFR [$M_\odot$ yr$^{-1}$]')
ax_sfh.xaxis.set_major_locator(plt.MaxNLocator(3))
ax_sfh.yaxis.set_major_locator(plt.LogLocator(numticks=3))

# ---------------------------------------------------------------------------
# Spectrum panel — optical
# ---------------------------------------------------------------------------
ax_spec_opt.set_xlim(4500, 9250)
ax_spec_opt.set_xlabel(r'Wavelength $[\AA]$')
ax_spec_opt.set_ylabel(r'$\log_{10}\;$Flux [Jy]')
ax_spec_opt.xaxis.set_major_locator(plt.MaxNLocator(3))
ax_spec_opt.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_spec_opt.tick_params(labelsize=FONTSIZE)
ax_spec_opt.spines['right'].set_visible(False)
ax_spec_opt.tick_params(right=False)

# ---------------------------------------------------------------------------
# Spectrum panel — IR  (both photometry points AND posterior model spectrum)
# ---------------------------------------------------------------------------
# Expected external variables (you supply these, analogous to sfh_samples):
#   ir_wave        : 1-D array of wavelengths in microns covering IRAC bands
#   spec_samples_ir: 2-D array [n_draw, len(ir_wave)] of posterior model fluxes in maggies

irac_x      = [3.6, 4.5]
irac_labels = [r'$3.6$', r'$4.5$']

# ax_spec_ir.set_xlim(0, 3)
ax_spec_ir.set_xticks([])                   # will be set explicitly after plotting
ax_spec_ir.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_spec_ir.tick_params(left=False, labelleft=False)
ax_spec_ir.spines['left'].set_visible(False)
ax_spec_ir.set_xlabel('Photometry')

# --- posterior model spectrum in IR band ---
ir_spec_lo  = np.percentile(spec_samples_ir, 16, axis=0)
ir_spec_med = np.percentile(spec_samples_ir, 50, axis=0)
ir_spec_hi  = np.percentile(spec_samples_ir, 84, axis=0)

ax_spec_ir.fill_between(
    ir_wave,
    np.log10(ir_spec_lo  * 3631),
    np.log10(ir_spec_hi  * 3631),
    alpha=0.3, color='#77aca2', zorder=1,
)
ax_spec_ir.plot(
    ir_wave,
    np.log10(ir_spec_med * 3631),
    color='#77aca2', linewidth=1.2, zorder=2,
)

# --- observed photometry on top ---
ax_spec_ir.errorbar(
    irac_x,
    np.log10(observations_dict['maggies'] * 3631),
    yerr=0.432 * observations_dict['maggies_unc'] * 6 / observations_dict['maggies'],
    fmt='s', color='#ff004f', capsize=3, ecolor='gray', zorder=5,
)

# restore the micron tick labels after all plotting is done
ax_spec_ir.set_xticks(irac_x)
ax_spec_ir.set_xticklabels(irac_labels)
ax_spec_ir.xaxis.set_tick_params(length=4)

# --- optical observed spectrum + uncertainty envelope ---
ax_spec_opt.plot(
    observations_dict['wavelength'],
    np.log10(observations_dict['spectrum'] * 3631),
    color='#ff004f', linewidth=1.5, alpha=0.7, zorder=10,
)
ax_spec_opt.fill_between(
    observations_dict['wavelength'],
    0.99 * np.log10(spec_lo * 3631),
    1.01 * np.log10(spec_hi * 3631),
    alpha=0.3, color='#77aca2',
)

# ---------------------------------------------------------------------------
# Subfigure margins
# ---------------------------------------------------------------------------
subfigs[0].subplots_adjust(left=0.08, right=0.98)
spec_subfigs[0].subplots_adjust(left=0.1,  right=0.95)
spec_subfigs[1].subplots_adjust(left=0.05, right=0.95)

add_break_markers(ax_spec_opt, ax_spec_ir)

# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------
corner.corner(
    df,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": FONTSIZE},
    label_kwargs={"fontsize": FONTSIZE},
    labelpad=0.15,
    bins=25,
    plot_datapoints=False,
    fig=subfigs[2],
    color='k',
)

corner_axes = np.array(subfigs[2].axes).reshape((n_params, n_params))

for ax in subfigs[2].axes:
    ax.tick_params(labelsize=FONTSIZE)
    if ax.get_title():
        ax.title.set_fontsize(FONTSIZE)
    ax.xaxis.label.set_fontsize(FONTSIZE)
    ax.yaxis.label.set_fontsize(FONTSIZE)

# ---------------------------------------------------------------------------
# Per-axis: precision-matched titles + prior overlays
# ---------------------------------------------------------------------------
for i in range(n_params):
    ax_diag  = corner_axes[i, i]
    samples  = df.iloc[:, i].values
    q16, q50, q84 = np.percentile(samples, [16, 50, 84])

    title_str = format_title(q50, q50 - q16, q84 - q50)   # ← new signature
    ax_diag.set_title(
        f'{labels[i].strip()}\n{title_str}',
        fontsize=FONTSIZE, pad=3,
    )

    key = param_keys[i]
    if key is not None:
        try:
            param_cfg = next(p for p in model.config_list if p['name'] == key)
            prior_obj = param_cfg.get('prior', None)
            if prior_obj is not None:
                pad_frac = 0.15 * (samples.max() - samples.min())
                x_prior  = np.linspace(samples.min() - pad_frac,
                                       samples.max() + pad_frac, 300)
                log_p  = prior_obj(x_prior)
                p_vals = np.exp(log_p - log_p.max())
                hist_vals, _ = np.histogram(samples, bins=25)
                scale    = hist_vals.max()
                ax_twin  = ax_diag.twinx()
                ax_twin.plot(x_prior, p_vals * scale,
                             color='darkorange', linewidth=1.2,
                             linestyle='--', alpha=0.8)
                ax_twin.set_ylim(0, scale * 1.1)
                ax_twin.set_yticks([])
        except (StopIteration, Exception):
            pass

    ax_diag.xaxis.set_major_locator(plt.MaxNLocator(3))

# ---------------------------------------------------------------------------
# Off-diagonal axes: tick locators + nudge bottom-row x labels downward
# ---------------------------------------------------------------------------
LABEL_NUDGE = 15   # points; increase if you want more separation

for i in range(n_params):
    for j in range(i):
        ax = corner_axes[i, j]
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        # bottom row only: shift the x-axis label down
        if i == n_params - 1:
            ax.xaxis.labelpad = LABEL_NUDGE

# also nudge the diagonal bottom cell's x label
corner_axes[n_params - 1, n_params - 1].xaxis.labelpad = LABEL_NUDGE

plt.savefig('corner_a4.pdf', bbox_inches='tight', dpi=300,
            facecolor='white', format='pdf')
plt.show()