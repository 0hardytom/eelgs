import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

def calculate_cluster_metrics(tbl):
    """
    Calculates LoS velocity, angular separation, projected distance, and 3D distance from a cluster center.

    Parameters:
    tbl (astropy.table.Table): Table with columns 'ra', 'dec', 'cluster_ra', 'cluster_dec', 'z', 'zcluster'.

    Returns:
    astropy.table.Table: Table with new columns 'los_velocity_kms', 'angular_separation_arcsec', 
                         'projected_distance_mpc', and 'distance_3d_mpc'.
    """
    c = 299792.458  # Speed of light in km/s

    # Calculate Line of Sight (LoS) velocity
    tbl['los_velocity_kms'] = c * (tbl['z'] - tbl['zcluster']) / (1 + tbl['zcluster'])
    tbl['los_velocity_kms'].unit = 'km/s'

    # Calculate angular separation
    galaxy_coords = SkyCoord(tbl['ra'], tbl['dec'], unit='deg', frame='icrs')
    cluster_coords = SkyCoord(tbl['cluster_ra'], tbl['cluster_dec'], unit='deg', frame='icrs')
    angular_separation = galaxy_coords.separation(cluster_coords)
    tbl['angular_separation_arcsec'] = angular_separation.arcsec
    tbl['angular_separation_arcsec'].unit = 'arcsec'

    # Calculate projected physical distance
    # Using the angular diameter distance to the cluster
    D_A = cosmo.angular_diameter_distance(tbl['zcluster'])
    # Small angle approximation: distance = angular_separation_in_radians * D_A
    tbl['projected_distance_mpc'] = (angular_separation.radian * D_A).to(u.Mpc)
    tbl['projected_distance_mpc'].unit = 'Mpc'

    # Calculate 3D distance
    d_c = cosmo.comoving_distance(tbl['zcluster'])
    d_g = cosmo.comoving_distance(tbl['z'])
    
    # Law of cosines
    distance_3d_sq = d_c**2 + d_g**2 - 2 * d_c * d_g * np.cos(angular_separation.radian)
    tbl['distance_3d_mpc'] = np.sqrt(distance_3d_sq.value) * u.Mpc
    tbl['distance_3d_mpc'].unit = 'Mpc'

    return tbl

if __name__ == '__main__':
    # Load the data
    try:
        master_tbl = Table.read('MASTER_new.csv', format='csv')
    except FileNotFoundError:
        print("Error: MASTER_new.csv not found. Please ensure the file is in the correct directory.")
        exit()

    # Calculate the new metrics
    master_tbl_with_metrics = calculate_cluster_metrics(master_tbl)

    # Save the new table to a csv file
    master_tbl_with_metrics.write('MASTER_with_metrics.csv', format='csv', overwrite=True)

    print("Calculations complete. New file saved as 'MASTER_with_metrics.csv'")
    print(master_tbl_with_metrics)
