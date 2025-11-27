
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

def crossmatch_vizier(galaxy_table):
    """
    Crossmatches an Astropy table of galaxy coordinates with a Vizier catalogue.

    Args:
        galaxy_table (astropy.table.Table): An Astropy table containing galaxy data.
            Must have 'ra' and 'dec' columns in degrees.

    Returns:
        astropy.table.Table: A table of the crossmatched sources from Vizier.
    """
    # Define the Vizier catalogue and columns to retrieve
    vizier_catalog = "J/ApJS/214/24/3dhstall"
    
    # Set up Vizier query object
    v = Vizier(columns=['*'])
    v.ROW_LIMIT = -1  # No row limit

    # Prepare coordinates from the input table
    coords = SkyCoord(ra=galaxy_table['ra'], dec=galaxy_table['dec'], unit=(u.deg, u.deg))

    # Perform the crossmatch
    result = v.query_region(coords, radius=1 * u.arcsec, catalog=vizier_catalog)

    if result:
        return result[0]
    else:
        return Table()

if __name__ == '__main__':
    # This is an example of how to use the function.
    # Replace this with your actual Astropy table.
    
    # Create a dummy table for demonstration purposes
    my_galaxies = Table()
    my_galaxies['ra'] = [150.119, 214.825]
    my_galaxies['dec'] = [2.224, 52.671]

    # Perform the crossmatch
    crossmatched_table = crossmatch_vizier(my_galaxies)

    # Print the results
    if len(crossmatched_table) > 0:
        print("Crossmatch Results:")
        print(crossmatched_table)
    else:
        print("No crossmatches found.")
