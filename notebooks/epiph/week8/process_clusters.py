
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad
import sys

def get_cluster_coords(cluster_name):
    """
    Queries SIMBAD for the coordinates of a cluster.
    """
    custom_simbad = Simbad()
    try:
        result_table = custom_simbad.query_object(cluster_name)
        if result_table is None or len(result_table) == 0:
            print(f"No results found for cluster: {cluster_name}", file=sys.stderr)
            return None, None
        
        # Check if RA and DEC columns exist
        if 'RA' not in result_table.colnames or 'DEC' not in result_table.colnames:
            print(f"Coordinates not found for cluster: {cluster_name}", file=sys.stderr)
            return None, None

        ra = result_table['RA'][0]
        dec = result_table['DEC'][0]
        return ra, dec
    except Exception as e:
        print(f"An error occurred while querying for {cluster_name}: {e}", file=sys.stderr)
        return None, None

def calculate_delta_r(galaxy_ra, galaxy_dec, cluster_ra, cluster_dec):
    """
    Calculates the angular separation between a galaxy and a cluster center.
    """
    try:
        galaxy_coord = SkyCoord(ra=galaxy_ra*u.degree, dec=galaxy_dec*u.degree, frame='icrs')
        cluster_coord = SkyCoord(ra=cluster_ra, dec=cluster_dec, unit=(u.hourangle, u.deg), frame='icrs')
        delta_r = galaxy_coord.separation(cluster_coord).arcmin
        return delta_r
    except Exception as e:
        print(f"Could not calculate separation for ra={galaxy_ra}, dec={galaxy_dec}: {e}", file=sys.stderr)
        return None

def main():
    # Read the csv file
    df = pd.read_csv('allsources.csv')

    # Get unique cluster names
    cluster_names = df['name'].unique()

    # Get cluster coordinates
    cluster_coords = {}
    for name in cluster_names:
        ra, dec = get_cluster_coords(name)
        if ra is not None:
            cluster_coords[name] = (ra, dec)

    # Calculate delta R for each galaxy
    df['delta_R_arcmin'] = df.apply(
        lambda row: calculate_delta_r(
            row['ra'],
            row['dec'],
            cluster_coords[row['name']][0],
            cluster_coords[row['name']][1]
        ) if row['name'] in cluster_coords else None,
        axis=1
    )

    # Save the results
    df.to_csv('allsources_with_delta_r.csv', index=False)
    print("Processing complete. Results saved to 'allsources_with_delta_r.csv'")

if __name__ == '__main__':
    main()
