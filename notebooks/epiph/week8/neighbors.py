import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import ascii

def count_neighbors(target_galaxies, parent_sample, search_radius_mpc, z_window):
    """
    Counts the number of neighboring galaxies for each target galaxy.

    Parameters:
    - target_galaxies (astropy.table.Table): Table with 'RA', 'DEC', 'Z' for the target sample.
    - parent_sample (astropy.table.Table): Table with 'RA', 'DEC', 'Z' for the parent sample.
    - search_radius_mpc (float): The physical search radius in Megaparsecs at the redshift of the target.
    - z_window (float): The half-width of the redshift slice to search within (e.g., 0.01).

    Returns:
    - numpy.ndarray: An array containing the number of neighbors for each target galaxy.
    """
    neighbor_counts = []

    # Create SkyCoord objects for the entire parent sample for efficient matching
    parent_coords = SkyCoord(parent_sample['RA']*u.deg, parent_sample['DEC']*u.deg, frame='icrs')

    for target in target_galaxies:
        target_z = target['Z']
        target_coord = SkyCoord(target['RA']*u.deg, target['DEC']*u.deg, frame='icrs')

        # --- Step 1: Redshift Slice ---
        # Select galaxies from the parent sample within the redshift window
        z_min = target_z - z_window
        z_max = target_z + z_window
        in_z_slice = (parent_sample['Z'] > z_min) & (parent_sample['Z'] < z_max)
        
        # --- Step 2: Angular Search ---
        # Calculate the angular separation on the sky for galaxies in the redshift slice
        separations = target_coord.separation(parent_coords[in_z_slice])

        # --- Step 3: Count Neighbors ---
        # Count how many of those galaxies are within the search radius
        # Note: We subtract 1 to exclude the target galaxy itself from its own neighbor count
        count = np.sum(separations < separations.max()) - 1
        neighbor_counts.append(count)

    return np.array(neighbor_counts)

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create some dummy data for demonstration
    # In your real use case, you would load your CSV files here, e.g.:
    # target_galaxies = Table.read('oii_lya_candidates.csv')
    # parent_sample = Table.read('allsources.csv')

    # A target sample of 5 galaxies
    # target_data = {
    #     'RA': [150.1, 150.2, 150.3, 150.4, 150.5],
    #     'DEC': [2.1, 2.2, 2.3, 2.4, 2.5],
    #     'Z': [1.0, 1.05, 1.1, 1.15, 1.2]
    # }
    # target_galaxies = Table(target_data)

    # A larger parent sample of 20 galaxies (including the targets)
    # parent_data = {
    #     'RA': [150.1, 150.11, 150.12, 150.2, 150.21, 150.3, 150.31, 150.32, 150.33, 150.4, 150.5, 150.51, 150.52, 150.53, 150.54, 149.9, 149.8, 151.0, 151.1, 151.2],
    #     'DEC': [2.1, 2.11, 2.12, 2.2, 2.21, 2.3, 2.31, 2.32, 2.33, 2.4, 2.5, 2.51, 2.52, 2.53, 2.54, 2.0, 1.9, 2.8, 2.9, 3.0],
    #     'Z': [1.0, 1.01, 0.99, 1.05, 1.06, 1.1, 1.11, 1.09, 1.12, 1.15, 1.2, 1.21, 1.19, 1.22, 1.18, 1.0, 1.5, 1.1, 1.2, 0.8]
    # }
    # parent_sample = Table(parent_data)

    parent_sample = ascii.read('../week5/final_table.csv')
    target_data = parent_sample[parent_sample['WFoiii5007_ew']>100]


    # 2. Set search parameters
    SEARCH_RADIUS_DEG = 0.05  # Search radius in degrees (approx 1.2 arcmin)
    REDSHIFT_WINDOW = 0.01    # Redshift slice is z_target +/- 0.01

    # 3. Run the analysis
    counts = count_neighbors(target_data, parent_sample, SEARCH_RADIUS_DEG, REDSHIFT_WINDOW)

    print("Neighbor counts for each target galaxy:")
    print(counts)

    # 4. Create the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use integer bins for the histogram
    bins = np.arange(counts.max() + 2) - 0.5
    ax.hist(counts, bins=bins, edgecolor='black', alpha=0.7)

    ax.set_xlabel('Number of Neighbors (N)', fontsize=14)
    ax.set_ylabel('Number of Target Galaxies', fontsize=14)
    ax.set_title('Distribution of Galaxy Environments', fontsize=16)
    ax.set_xticks(np.arange(counts.max() + 1)) # Ensure integer ticks on x-axis
    plt.show()

    # 5. Create the Neighbor Count vs. Redshift plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(target_data['Z'], counts, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel('Number of Neighbors (N)', fontsize=14)
    ax.set_title('Galaxy Environment vs. Redshift', fontsize=16)
    ax.grid(True)
    plt.show()
