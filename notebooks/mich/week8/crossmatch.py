import numpy as np
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
from astropy import units as u

def crossmatch_tables(tab, final_table, separation_threshold=1.0 * u.arcsec):
    """
    Crossmatches two astropy tables based on RA and DEC coordinates.

    For each row in `tab`, it finds the closest match in `final_table` within
    a given separation threshold. If a match is found, columns from the 7th
    column onwards from `final_table` are appended to the row in `tab`.
    If no match is found, the new columns are filled with masked values (NaNs).

    Parameters
    ----------
    tab : astropy.table.Table
        The primary table with candidate galaxies. Must contain 'RA' and 'DEC'
        columns, assumed to be in degrees.
    final_table : astropy.table.Table
        The master photometry table to match against. Must contain 'RA' and 'DEC'
        columns, assumed to be in degrees.
    separation_threshold : astropy.units.Quantity
        The maximum separation for a match to be considered valid.
        Default is 1 arcsecond.

    Returns
    -------
    astropy.table.Table
        The `tab` table with appended columns from `final_table` for matched sources.
    """
    coords_tab = SkyCoord(ra=tab['ra']*u.degree, dec=tab['dec']*u.degree)
    coords_final = SkyCoord(ra=final_table['RA']*u.degree, dec=final_table['DEC']*u.degree)
    idx, d2d, d3d = coords_tab.match_to_catalog_sky(coords_final)
    good_match_mask = d2d <= separation_threshold
    cols_to_add_names = final_table.colnames[6:]
    matched_photometry = final_table[cols_to_add_names][idx]
    final_appended_cols = Table(matched_photometry, masked=True)
    for col_name in final_appended_cols.colnames:
        final_appended_cols[col_name].mask[~good_match_mask] = True
    result_table = hstack([tab, final_appended_cols])

    return result_table
