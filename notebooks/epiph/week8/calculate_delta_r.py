import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astroquery.vizier import Vizier
import numpy as np
import sys

def get_cluster_coords(cluster_df, cluster_name):
    """
    Finds cluster coordinates by taking the mean position of its members
    and querying for the nearest cluster in Simbad. This is more robust
    than relying on name resolution for shorthand cluster names.
    """
    if cluster_df.empty:
        return None

    mean_ra = cluster_df['ra'].mean()
    mean_dec = cluster_df['dec'].mean()
    
    mean_coord = SkyCoord(ra=mean_ra*u.degree, dec=mean_dec*u.degree, frame='icrs')
    
    try:
        # Query Simbad for clusters of galaxies near the mean position of the ELGs
        # A 10 arcminute radius should be sufficient to find the cluster center
        result_table = Vizier.query_region(mean_coord, radius=10*u.arcmin, catalog="simbad", otype='ClG')
        
        if not result_table or len(result_table[0]) == 0:
            print(f"\nWarning: No cluster found for '{cluster_name}' near mean position RA={mean_ra:.4f}, Dec={mean_dec:.4f}. Trying a wider search for any object.")
            # Fallback to a wider search without the object type constraint if no cluster is found
            result_table = Vizier.query_region(mean_coord, radius=2*u.arcmin, catalog="simbad")
            if not result_table or len(result_table[0]) == 0:
                print(f"\nWarning: No astronomical object found for '{cluster_name}' even with a wider search.")
                return None

        # Find the closest match from the results to the mean coordinate
        results_df = result_table[0].to_pandas()
        result_coords = SkyCoord(ra=results_df['RA'], dec=results_df['DEC'], unit=(u.hourangle, u.deg), frame='icrs')
        
        separations = mean_coord.separation(result_coords)
        closest_idx = np.argmin(separations)
        
        closest_cluster = results_df.iloc[closest_idx]
        
        coord_str = f"{closest_cluster['RA']} {closest_cluster['DEC']}"
        # print(f"\nFound '{closest_cluster['MAIN_ID']}' for cluster '{cluster_name}'.")
        return SkyCoord(coord_str, unit=(u.hourangle, u.deg), frame='icrs')

    except Exception as e:
        print(f"\nError querying for '{cluster_name}' by position: {e}")
        return None

def calculate_delta_r(elg_ra, elg_dec, elg_z, cluster_coord):
    """
    Calculates the projected physical separation (delta R) in Mpc between an ELG and its cluster center.
    """
    if cluster_coord is None or pd.isna(elg_z) or elg_z <= 0:
        return np.nan

    try:
        elg_coord = SkyCoord(ra=elg_ra*u.degree, dec=elg_dec*u.degree, frame='icrs')
        
        # Calculate angular separation on the sky
        angular_sep = elg_coord.separation(cluster_coord)

        # Get the angular diameter distance from the cosmology model for the given redshift
        ang_diam_dist = cosmo.angular_diameter_distance(elg_z)

        # Calculate the projected physical distance
        delta_r = (angular_sep.to(u.rad).value * ang_diam_dist).to(u.Mpc)

        return delta_r.value
    except Exception as e:
        print(f"\nCould not calculate delta_R for ra={elg_ra}, dec={elg_dec}, z={elg_z}: {e}")
        return np.nan

def main():
    """
    Main function to execute the script.
    """
    input_file = 'allsources.csv'
    output_file = 'allsources_with_delta_r.csv'

    print(f"Reading ELG data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return

    required_cols = ['name', 'ra', 'dec', 'z']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input file must contain columns: {', '.join(required_cols)}")
        return

    unique_clusters = df['name'].unique()
    print(f"Found {len(unique_clusters)} unique clusters.")

    print("Querying Vizier for cluster coordinates...")
    cluster_coords = {}
    for i, cluster_name in enumerate(unique_clusters):
        sys.stdout.write(f"\rProcessing cluster {i+1}/{len(unique_clusters)}: {cluster_name.ljust(20)}")
        sys.stdout.flush()
        
        if cluster_name == 'STACK' or pd.isna(cluster_name):
            cluster_coords[cluster_name] = None
            continue
            
        cluster_df = df[df['name'] == cluster_name].copy()
        cluster_coords[cluster_name] = get_cluster_coords(cluster_df, cluster_name)
    print("\nFinished querying for cluster coordinates.")

    print("Calculating projected distances (delta R)...")
    df['delta_R_Mpc'] = df.apply(
        lambda row: calculate_delta_r(
            row['ra'],
            row['dec'],
            row['z'],
            cluster_coords.get(row['name'])
        ),
        axis=1
    )

    print(f"Saving results to '{output_file}'...")
    df.to_csv(output_file, index=False)
    print("Script finished successfully.")

if __name__ == '__main__':
    main()