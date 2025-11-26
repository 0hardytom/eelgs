from astropy.table import Table, join
from astroquery.vizier import VizieR

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