import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS

def get_sdss_solar_masses(galaxy_table):
    """
    Queries SDSS for solar masses for a table of galaxies.

    This function cross-matches galaxies from the input table with the SDSS database
    and retrieves their stellar mass from the MPA-JHU catalog (GalSpecInfo).

    Parameters
    ----------
    galaxy_table : astropy.table.Table
        Table containing galaxy coordinates. Must have 'ra' and 'dec'
        columns in degrees.

    Returns
    -------
    astropy.table.Table
        The input table with an added 'solar_mass' column.
        The mass is in solar mass units. Values are None if no match is found.
    """
    solar_masses = []
    
    print(f"Querying SDSS for {len(galaxy_table)} galaxies...")
    
    for i, galaxy in enumerate(galaxy_table):
        ra = galaxy['ra']
        dec = galaxy['dec']
        
        print(f"  ({i+1}/{len(galaxy_table)}) Querying for RA={ra:.4f}, Dec={dec:.4f}...")
        
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
        
        # Search radius for the cross-match
        radius = 2 * u.arcsec
        
        try:
            # Cross-match to get the spectral object ID
            xid = SDSS.query_crossid(coords, radius=radius, specobj_fields=['specObjID'])
            
            if xid:
                specobjid = xid['specObjID'][0]
                
                # Query the MPA-JHU catalog for the stellar mass
                mass_query = f"SELECT stellarMass FROM GalSpecInfo WHERE specObjID = {specobjid}"
                mass_result = SDSS.query_sql(mass_query)
                
                if mass_result:
                    # The stellar mass is given in log10(M_sun)
                    log_mass = mass_result['stellarMass'][0]
                    solar_masses.append(10**log_mass)
                else:
                    solar_masses.append(None)
            else:
                solar_masses.append(None)
        except Exception as e:
            print(f"An error occurred for galaxy at RA={ra}, Dec={dec}: {e}")
            solar_masses.append(None)
            
    galaxy_table['solar_mass'] = solar_masses
    galaxy_table['solar_mass'].unit = u.solMass
    
    return galaxy_table

# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy table similar to yours
    # Replace this with your actual data
    my_galaxies = Table()
    my_galaxies['ra'] = [146.71, 146.75, 150.12] * u.deg
    my_galaxies['dec'] = [-1.05, -1.08, 2.20] * u.deg

    # Get the solar masses
    my_galaxies_with_masses = get_sdss_solar_masses(my_galaxies)

    # Print the result
    print("\n--- Results ---")
    print(my_galaxies_with_masses)
