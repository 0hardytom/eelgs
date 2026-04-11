import numpy as np
import matplotlib.pyplot as plt

def plot_sfh_step_continuous(ax, agebins, sfh_lo, sfh_hi, sfh_med):
    """
    Plots star formation history as a continuous step plot, correctly
    handling non-contiguous bins.

    This version builds the coordinate arrays for plotting and explicitly
    inserts `np.nan` where gaps between bins occur. This prevents matplotlib
    from drawing incorrect connecting lines across those gaps.

    Parameters:
    - ax: The matplotlib axes object to plot on.
    - agebins: An array of shape (N, 2) defining the age bins in Gyr.
    - sfh_lo: The lower uncertainty of the SFH for each bin.
    - sfh_hi: The upper uncertainty of the SFH for each bin.
    - sfh_med: The median SFH for each bin.
    """
    bin_lo_gyr = agebins[:, 0]
    bin_hi_gyr = agebins[:, 1]

    # Create coordinate arrays for a continuous step plot.
    # We start by creating the coordinates as if all bins were contiguous.
    # x will be [bin_lo_0, bin_hi_0, bin_lo_1, bin_hi_1, ...]
    # y will be [sfh_0,    sfh_0,    sfh_1,    sfh_1,    ...]
    x_coords = list(np.array([bin_lo_gyr, bin_hi_gyr]).T.flatten())
    y_med_coords = list(np.repeat(sfh_med, 2))
    y_lo_coords = list(np.repeat(sfh_lo, 2))
    y_hi_coords = list(np.repeat(sfh_hi, 2))

    # Now, iterate through the bins to find any gaps.
    # Where a gap exists, we insert a `NaN` to break the line plot.
    insert_offset = 0
    for i in range(agebins.shape[0] - 1):
        # Check if the end of the current bin matches the start of the next one
        if bin_hi_gyr[i] != bin_lo_gyr[i+1]:
            # If they don't match, there's a gap.
            # We need to insert a break after the points for the current bin.
            # The insertion index is after the two points of the current bin.
            insert_idx = 2 * (i + 1) + insert_offset
            
            x_coords.insert(insert_idx, np.nan)
            y_med_coords.insert(insert_idx, np.nan)
            y_lo_coords.insert(insert_idx, np.nan)
            y_hi_coords.insert(insert_idx, np.nan)
            
            # Increment the offset since we've added an element to the lists
            insert_offset += 1
    x_coords = np.array(x_coords)

    # Plot the uncertainty range as a continuous filled area
    ax.fill_between(
        10**x_coords,
        y_lo_coords,
        y_hi_coords,
        alpha=0.3, color='#77aca2',
        label='68% Confidence Interval'
    )

    # Plot the median SFH as a continuous step line
    ax.plot(
        10**x_coords,
        y_med_coords,
        color='#ff004f',
        linewidth=3,
        label='Median SFH'
    )

# --- Example Usage ---
if __name__ == '__main__':
    # --- Case 1: Contiguous Bins ---
    bin_edges = np.logspace(-1, 1, 9)
    contiguous_agebins = np.array([bin_edges[:-1], bin_edges[1:]]).T
    sfh_med_1 = 10 * np.exp(-bin_edges[:-1]) + np.random.uniform(0, 2, size=8)
    sfh_error_1 = sfh_med_1 * 0.2
    sfh_lo_1 = sfh_med_1 - sfh_error_1
    sfh_hi_1 = sfh_med_1 + sfh_error_1

    # --- Case 2: Non-Contiguous Bins (with a gap) ---
    # Create a gap between the 4th and 5th bins
    non_contiguous_agebins = np.copy(contiguous_agebins)
    non_contiguous_agebins[4:, 0] += 1.5 # Shift the start of later bins
    non_contiguous_agebins[4:, 1] += 1.5 # Shift the end of later bins

    # Create a figure with two subplots to show both cases
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 14), sharey=True
    )

    # Plot the contiguous case
    plot_sfh_step_continuous(ax1, contiguous_agebins, sfh_lo_1, sfh_hi_1, sfh_med_1)
    ax1.set_title("Corrected Plot with Contiguous Bins", fontsize=16)
    ax1.set_xlabel("Lookback Time (Gyr)", fontsize=14)
    ax1.set_ylabel("Star Formation Rate (M$_\odot$/yr)", fontsize=14)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_xlim(bin_edges[0], np.max(contiguous_agebins) * 1.1)
    ax1.set_ylim(0, np.max(sfh_hi_1) * 1.1)

    # Plot the non-contiguous case
    plot_sfh_step_continuous(ax2, non_contiguous_agebins, sfh_lo_1, sfh_hi_1, sfh_med_1)
    ax2.set_title("Corrected Plot with a Gap Between Bins", fontsize=16)
    ax2.set_xlabel("Lookback Time (Gyr)", fontsize=14)
    ax2.set_ylabel("Star Formation Rate (M$_\odot$/yr)", fontsize=14)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()
    # Using a linear scale here to make the gap obvious
    ax2.set_xlim(0, np.max(non_contiguous_agebins) * 1.1)
    
    plt.tight_layout()
    
    # Save the plot to a file
    output_filename = "corrected_sfh_step_plot.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    
    print(f"Corrected function saved to step.py")
    print(f"Example plot demonstrating both cases saved to {output_filename}")