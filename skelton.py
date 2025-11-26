from astropy.table import Table, join
from astroquery.vizier import VizieR
import numpy as np

# --- 1. Your Input Data ---
# Let's assume you have your redshift catalog in an Astropy Table.
# For this example, I'll create a dummy Table.
# In your real use case, you would load your FITS or ECSV file here.
# For example:
# your_data = Table.read('path/to/your/redshift_catalog.fits')

your_data = Table({
    'UNIQUE_ID': [101, 102, 103],
    'SKELTON_ID': [18284, 18319, 18422], # Example IDs from the GOODS-S field in Skelton+14
    'LEAD_LINE': ['O2', 'Ha', 'O3_2'],
    # ... other columns from your redshift table
})

# --- 2. Configure the VizieR Query ---
# Set up the VizieR query tool
v = VizieR(
    columns=['ID', 'RA', 'DEC', 
             'f_F435W', 'e_F435W', # B-band
             'f_F606W', 'e_F606W', # V-band
             'f_F814W', 'e_F814W', # I-band
             'f_F125W', 'e_F125W', # J-band
             'f_F140W', 'e_F140W', # H-band (intermediate)
             'f_F160W', 'e_F160W'  # H-band
            ],
    row_limit=-1 # Get all matching rows
)

# The catalog name for Skelton et al. 2014 (3D-HST)
# This corresponds to the main catalog file.
catalog_id = 'J/ApJS/214/24/catalog'

# Get the list of IDs to query from your table
skelton_ids_to_query = your_data['SKELTON_ID']

# --- 3. Perform the Query ---
print(f"Querying VizieR for {len(skelton_ids_to_query)} objects from catalog {catalog_id}...")

# Construct a single query string for efficiency. 
# This asks VizieR for rows where the ID is one of the numbers in our list.
id_query_string = ' || '.join([f"=={id_num}" for id_num in skelton_ids_to_query])

try:
    # The result is a list of tables; we want the first one.
    result_tables = v.query_constraints(
        catalog=catalog_id,
        ID=id_query_string
    )
    if not result_tables:
        raise ValueError("Query returned no tables. Check catalog ID and object IDs.")
    
    photometry_table = result_tables[0]
    
    print("Query complete.")
    print("Downloaded Photometry:")
    print(photometry_table)

    # --- 4. Merge and Finalize ---
    # Rename the 'ID' column to match your catalog for a clean merge
    photometry_table.rename_column('ID', 'SKELTON_ID')

    # Join the photometry back into your original table
    # 'join_type=left' ensures that all your original objects are kept, 
    # even if some don't have a match in the Skelton catalog.
    final_table = join(your_data, photometry_table, keys='SKELTON_ID', join_type='left')

    print("\nFinal Merged Table:")
    print(final_table)

    # You can now save this merged table to a new file
    # final_table.write('merged_catalog_with_photometry.fits', overwrite=True)

except Exception as e:
    print(f"An error occurred during the VizieR query: {e}")
    print("Please check your internet connection, the catalog identifier, and the SKELTON_IDs.")

def smooth_step(ax, x, y, **kwargs):
    """
    Plots a smooth, interpolated curve on a given matplotlib axes object,
    analogous to a smoothed version of ax.step().

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points.
    **kwargs : dict
        Additional keyword arguments to be passed to `ax.plot()`.
        Examples: 'color', 'linestyle', 'linewidth', 'label'.

    Returns
    -------
    list
        A list of the Line2D objects added to the axes.
    """
    import numpy as np
    from scipy.interpolate import make_interp_spline

    # Create a new, denser set of x-values for the smooth curve
    x_smooth = np.linspace(np.min(x), np.max(x), 300)

    # Create the spline interpolation function (cubic is a good default)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)

    # Plot the smoothed data on the provided axes
    line = ax.plot(x_smooth, y_smooth, **kwargs)
    
    return line

def add_oiii_fluxes(maintab, linetab):
    """
    Adds OIII 5007 and 4959 fluxes, errors, and centroids to a main table.

    This function takes a main catalog table and a line flux table and adds
    columns for the OIII 5007 and 4959 fluxes, errors, and observed
    wavelengths (centroids). The cross-matching is done using the 'UNIQUE_ID'
    column, which must be present in both tables.

    Parameters
    ----------
    maintab : astropy.table.Table
        The main table with general object properties. Must contain 'UNIQUE_ID'.
    linetab : astropy.table.Table
        The table with individual line measurements. Must contain 'UNIQUE_ID',
        'IDENT', 'F_3KRON', 'F_3KRON_ERR', and 'LAM_OBS'.

    Returns
    -------
    astropy.table.Table
        A new table containing the columns from maintab plus the six new
        OIII-related columns:
        - 'oiii5007_flux'
        - 'oiii5007_flux_err'
        - 'oiii5007_centroid'
        - 'oiii4959_flux'
        - 'oiii4959_flux_err'
        - 'oiii4959_centroid'
    """
    # Ensure UNIQUE_ID is a common key
    if 'UNIQUE_ID' not in maintab.colnames or 'UNIQUE_ID' not in linetab.colnames:
        raise ValueError("Both tables must contain a 'UNIQUE_ID' column for cross-matching.")

    # --- Process OIII 5007 (O3_2) ---
    oiii_5007_lines = linetab[linetab['IDENT'] == 'O3_2']
    oiii_5007_to_join = oiii_5007_lines[['UNIQUE_ID', 'F_3KRON', 'F_3KRON_ERR', 'LAM_OBS']]
    oiii_5007_to_join.rename_column('F_3KRON', 'oiii5007_flux')
    oiii_5007_to_join.rename_column('F_3KRON_ERR', 'oiii5007_flux_err')
    oiii_5007_to_join.rename_column('LAM_OBS', 'oiii5007_centroid')

    # --- Process OIII 4959 (O3_1) ---
    oiii_4959_lines = linetab[linetab['IDENT'] == 'O3_1']
    oiii_4959_to_join = oiii_4959_lines[['UNIQUE_ID', 'F_3KRON', 'F_3KRON_ERR', 'LAM_OBS']]
    oiii_4959_to_join.rename_column('F_3KRON', 'oiii4959_flux')
    oiii_4959_to_join.rename_column('F_3KRON_ERR', 'oiii4959_flux_err')
    oiii_4959_to_join.rename_column('LAM_OBS', 'oiii4959_centroid')

    # --- Join the tables ---
    # Start with the main table
    merged_table = join(maintab, oiii_5007_to_join, keys='UNIQUE_ID', join_type='left')
    merged_table = join(merged_table, oiii_4959_to_join, keys='UNIQUE_ID', join_type='left')

    # --- Fill missing values ---
    # After a left join, non-matches will be represented by masked values.
    # It's good practice to fill these with a sensible default, like 0 or NaN.
    # For fluxes and errors, NaN is often a good choice.
    for col in ['oiii5007_flux', 'oiii5007_flux_err', 'oiii5007_centroid',
                'oiii4959_flux', 'oiii4959_flux_err', 'oiii4959_centroid']:
        if col in merged_table.colnames and hasattr(merged_table[col], 'filled'):
            merged_table[col] = merged_table[col].filled(np.nan)

    return merged_table