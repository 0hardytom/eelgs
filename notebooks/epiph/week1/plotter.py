import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# The original script used a custom library 'plotfancy'.
# We will replace its functions with standard matplotlib/seaborn calls.
# For example, pf.create_plot() is replaced by plt.subplots(),
# and pf.fix_plot() is replaced by direct styling or removed.


def plot_kde_contours(ax, data, x_col, y_col, fill, **kwargs):
    """
    Helper function to draw a 2D Kernel Density Estimate plot with contours.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        data (pd.DataFrame): The data to plot.
        x_col (str): The name of the column for the x-axis.
        y_col (str): The name of the column for the y-axis.
        fill (bool): Whether to draw a filled contour plot.
    """
    sns.kdeplot(
        data=data,
        x=x_col,
        y=y_col,
        fill=fill,
        ax=ax,
        **kwargs
    )


def create_inset_plot(parent_ax, data, x_col, y_col):
    """
    Creates and styles an inset plot on the parent axes.

    Args:
        parent_ax (matplotlib.axes.Axes): The main axes to draw the inset on.
        data (pd.DataFrame): The data for the plot.
        x_col (str): The name of the column for the x-axis.
        y_col (str): The name of the column for the y-axis.
    """
    # Create inset axes in the upper left corner
    ax_inset = inset_axes(parent_ax, width="40%", height="40%", loc='upper left', borderpad=1)

    # Subset the data for the inset plot
    subset_df = data[data[y_col] > 2]

    # Plot the subset on the inset axes
    plot_kde_contours(
        ax_inset, subset_df, x_col, y_col, fill=True,
        cmap='magma_r', thresh=0.05
    )
    plot_kde_contours(
        ax_inset, subset_df, x_col, y_col, fill=False,
        color='k', linewidths=0.5, thresh=0.05
    )

    # Set limits and style for the inset plot
    ax_inset.set_xlim(parent_ax.get_xlim())
    ax_inset.set_ylim([2, 3.4])
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    ax_inset.tick_params(axis='x', which='major', labelsize=8, labelbottom=False, bottom=False)
    ax_inset.tick_params(axis='y', which='major', labelsize=8)

    # Indicate the area of the inset on the main plot
    mark_inset(parent_ax, ax_inset, loc1=3, loc2=1, fc="none", ec="0.5")


def plot_mass_ew_density(df, x_col, y_col, output_filename):
    """
    Generates and saves a 2D kernel density plot of stellar mass vs. OIII EW.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for the x-axis (log stellar mass).
        y_col (str): Column name for the y-axis (log OIII EW).
        output_filename (str): Path to save the output plot.
    """
    # 1. Create the plot axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # 2. Create the 2D Kernel Density plot
    # First call for the filled colormap and colorbar
    plot_kde_contours(
        ax, df, x_col, y_col, fill=True,
        cmap='magma_r', cbar=True,
        cbar_kws={'label': 'Point Density', 'format': '%.2f'},
        thresh=0.05
    )

    # --- Manual Colorbar Modification ---
    if ax.collections:
        mappable = ax.collections[0]
        cbar = mappable.colorbar
        if cbar:
            current_ticks = cbar.get_ticks()
            cbar.set_ticks(current_ticks[::2])
    # --- End Modification ---

    # Second call for the contour lines
    plot_kde_contours(
        ax, df, x_col, y_col, fill=False,
        color='k', linewidths=0.5, thresh=0.05
    )

    # 3. Set labels and limits
    ax.set_ylabel(r'$\log_{10}\;$EW[OIII]')
    ax.set_xlabel(r'$\log_{10}\;\mathrm{M}_{\mathrm{stell}}$')
    ax.set_xlim([5.4, 11.3])
    ax.set_ylim([-0.9, 3.4])

    # 4. Add inset plot
    create_inset_plot(ax, df, x_col, y_col)

    # 5. Save and show the plot
    # The original script called pf.fix_plot(). We can add direct styling
    # here if needed, e.g., ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.show()


def main():
    """
    Main function to generate sample data and create the plot.
    """
    # --- Data Preparation ---
    # The original script assumes a pre-existing astropy Table 'peas_final'.
    # Here, we generate a sample pandas DataFrame for demonstration purposes.
    # In your workflow, you would replace this with your actual data loading.
    print("Generating sample data...")
    num_points = 500
    sample_data = {
        'logmstellar': np.random.normal(loc=8.5, scale=1.5, size=num_points),
        'oiii5007_ew': 10**np.random.normal(loc=1.5, scale=0.8, size=num_points)
    }
    peas_final_df = pd.DataFrame(sample_data)
    peas_final_df['log_oiii5007_ew'] = np.log10(peas_final_df['oiii5007_ew'])
    print("Sample data generated.")

    # Define columns and output file
    x_col = 'logmstellar'
    y_col = 'log_oiii5007_ew'
    output_file = 'figs/mass_ew_density_seaborn_cleaned.png'

    # Create the plot
    plot_mass_ew_density(peas_final_df, x_col, y_col, output_file)


if __name__ == "__main__":
    # Create figs directory if it doesn't exist
    if not os.path.exists('figs'):
        os.makedirs('figs')
    main()