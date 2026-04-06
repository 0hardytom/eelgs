import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotfancy as pf
import matplotlib.patches as patches

def plot_kde_with_contours(ax, data, x, y, cmap='magma_r', alpha=1.0, cbar=True):
    """
    Plots a 2D Kernel Density Estimate with contours on a given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        data (pd.DataFrame): The data to plot.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        cmap (str, optional): Colormap name. Defaults to 'magma_r'.
        alpha (float, optional): Alpha for the fill. Defaults to 1.0.
        cbar (bool, optional): Whether to add a colorbar. Defaults to True.
    """
    # First call: Draw the filled colormap and create the colorbar.
    cbar_kwargs = {'label': 'Density', 'format': '%.2f'} if cbar else {}
    sns.kdeplot(
        data=data,
        x=x,
        y=y,
        fill=True,
        cmap=cmap,
        alpha=alpha,
        ax=ax,
        cbar=cbar,
        cbar_kws=cbar_kwargs,
        thresh=0.05
    )

    # --- Manual Colorbar Modification ---
    if cbar and ax.collections:
        mappable = ax.collections[0]
        cbar_obj = mappable.colorbar
        if cbar_obj:
            current_ticks = cbar_obj.get_ticks()
            cbar_obj.set_ticks(current_ticks[::2])
    # --- End Modification ---

    # Second call: Overlay the contour lines in a different color.
    sns.kdeplot(
        data=data,
        x=x,
        y=y,
        fill=False,
        color='k',
        linewidths=0.5,
        ax=ax,
        thresh=0.05
    )

def create_mass_ew_density_plot(peas_final, output_filename):
    """
    Creates and saves the mass vs. EW[OIII] density plot.

    Args:
        peas_final: vaex dataframe or similar object with .to_pandas() method.
        output_filename (str): The path to save the output plot.
    """
    fig, ax = pf.create_plot()

    peas_final_df = peas_final.to_pandas()
    peas_final_df['log_oiii5007_ew'] = np.log10(peas_final_df['oiii5007_ew'])

    plot_kde_with_contours(ax, peas_final_df, 'logmstellar', 'log_oiii5007_ew')
    ax.text(0.2, 0.90, 'ELGs', transform=ax.transAxes, fontsize=15, ha='right', weight='bold')

    ax.set_xlabel(r'$\log_{10}\;\mathrm{M}_{\mathrm{stell}}$', fontsize=15)

    ax.set_xlim([5.4, 11.3])
    ax.set_ylim([-0.9, 3.4])
    ax.fill_between([-100, 100], -5, 2, hatch='//', edgecolor='k', facecolor='none', zorder=-100)

    ax2 = fig.add_axes((0, 1.02, 1, 1), sharex=ax)
    plt.setp(ax2.get_xticklabels(), visible=False)

    eelgs_final_df = peas_final_df[peas_final_df['log_oiii5007_ew'] > 2]

    # Plot EELG KDE on the top plot (ax2) with cividis colormap and no transparency
    plot_kde_with_contours(ax2, eelgs_final_df, 'logmstellar', 'log_oiii5007_ew', cmap='cividis', alpha=1.0)
    ax2.text(0.05, 0.95, 'EELGs', transform=ax2.transAxes, fontsize=15, va='top', weight='bold')

    # Plot EELG KDE on the main plot (ax) as well, without a colorbar
    plot_kde_with_contours(ax, eelgs_final_df, 'logmstellar', 'log_oiii5007_ew', cmap='cividis', alpha=0.5, cbar=False)


    # Get bounding box from the contours of the KDE plot on ax2
    paths = []
    if len(ax2.collections) > 1:
        # The second collection created by plot_kde_with_contours is the line contours
        paths = ax2.collections[1].get_paths()

    if paths:
        # Calculate the bounding box that encloses all contour paths
        all_vertices = np.vstack([path.vertices for path in paths])
        x_min, y_min = all_vertices.min(axis=0)
        x_max, y_max = all_vertices.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
    else:
        # Fallback to data boundaries if contours are not available
        x_min = eelgs_final_df['logmstellar'].min()
        x_max = eelgs_final_df['logmstellar'].max()
        y_min = eelgs_final_df['log_oiii5007_ew'].min()
        y_max = eelgs_final_df['log_oiii5007_ew'].max()
        width = x_max - x_min
        height = y_max - y_min

    # Create and add the rectangle to ax2
    rect2 = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='k', facecolor='none', zorder=100)
    ax2.add_patch(rect2)

    # Create and add the same rectangle to ax
    rect1 = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='k', facecolor='none', zorder=100)
    ax.add_patch(rect1)

    pf.fix_plot([ax, ax2])

    fig.supylabel(r'$\log_{10}\;$EW[OIII]', position=(-0.15, 1))
    ax.set_ylabel('')
    ax2.set_ylabel('')

    ax2.set_ylim([1.9,3.49])
    ax2.set_yticks([2,2.5,3])

    fig.savefig(output_filename, dpi=600, bbox_inches='tight')


def create_mass_ew_density_plot2(peas_final, output_filename):
    """
    Creates and saves the mass vs. EW[OIII] density plot.

    Args:
        peas_final: vaex dataframe or similar object with .to_pandas() method.
        output_filename (str): The path to save the output plot.
    """
    fig, ax = pf.create_plot()

    peas_final_df = peas_final.to_pandas()
    peas_final_df['log_oiii5007_ew'] = np.log10(peas_final_df['oiii5007_ew'])

    plot_kde_with_contours(ax, peas_final_df, 'logmstellar', 'log_oiii5007_ew')
    ax.text(0.2, 0.90, 'ELGs', transform=ax.transAxes, fontsize=15, ha='right', weight='bold')

    ax.set_xlabel(r'$\log_{10}\;\mathrm{M}_{\mathrm{stell}}$', fontsize=15)

    ax.set_xlim([5.4, 11.3])
    ax.set_ylim([-0.9, 3.4])
    ax.fill_between([-100, 100], -5, 2, hatch='//', edgecolor='k', facecolor='none', zorder=-100)

    # ax2 = fig.add_axes((0, 1.02, 1, 1), sharex=ax)
    # plt.setp(ax2.get_xticklabels(), visible=False)

    eelgs_final_df = peas_final_df[peas_final_df['log_oiii5007_ew'] > 2]

    # Plot EELG KDE on the top plot (ax2) with cividis colormap and no transparency
    # plot_kde_with_contours(ax2, eelgs_final_df, 'logmstellar', 'log_oiii5007_ew', cmap='cividis', alpha=1.0)
    # ax2.text(0.05, 0.95, 'EELGs', transform=ax2.transAxes, fontsize=15, va='top', weight='bold')

    # Plot EELG KDE on the main plot (ax) as well, without a colorbar
    plot_kde_with_contours(ax, eelgs_final_df, 'logmstellar', 'log_oiii5007_ew', cmap='cividis', alpha=0.5, cbar=False)


    # Get bounding box from the contours of the KDE plot on ax2
    paths = []
    if len(ax2.collections) > 1:
        # The second collection created by plot_kde_with_contours is the line contours
        paths = ax2.collections[1].get_paths()

    if paths:
        # Calculate the bounding box that encloses all contour paths
        all_vertices = np.vstack([path.vertices for path in paths])
        x_min, y_min = all_vertices.min(axis=0)
        x_max, y_max = all_vertices.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
    else:
        # Fallback to data boundaries if contours are not available
        x_min = eelgs_final_df['logmstellar'].min()
        x_max = eelgs_final_df['logmstellar'].max()
        y_min = eelgs_final_df['log_oiii5007_ew'].min()
        y_max = eelgs_final_df['log_oiii5007_ew'].max()
        width = x_max - x_min
        height = y_max - y_min

    # Create and add the rectangle to ax2
    rect2 = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='k', facecolor='none', zorder=100)
    ax2.add_patch(rect2)

    # Create and add the same rectangle to ax
    rect1 = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='k', facecolor='none', zorder=100)
    ax.add_patch(rect1)

    pf.fix_plot([ax, ax2])

    fig.supylabel(r'$\log_{10}\;$EW[OIII]', position=(-0.15, 1))
    ax.set_ylabel('')
    ax2.set_ylabel('')

    ax2.set_ylim([1.9,3.49])
    ax2.set_yticks([2,2.5,3])

    fig.savefig(output_filename, dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    # This is an example of how to use the function.
    # You will need to load your data.
    # For example, if you have a vaex dataframe:
    # import vaex
    # peas_final = vaex.open('path/to/your/data.hdf5')
    # create_mass_ew_density_plot(peas_final, 'figs/mass_ew_density_seaborn_no_zero2.png')

    # If you have a pandas DataFrame, you might need to adjust the function
    # or wrap the dataframe in an object with a .to_pandas() method.
    # For demonstration purposes, this block is left empty.
    pass
