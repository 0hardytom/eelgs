
import pandas as pd
import os

# Get the current working directory
cwd = os.getcwd()
photometry_file = os.path.join(cwd, 'MUSE_photometry.csv')
output_file = os.path.join(cwd, 'missing_sources.csv')


# Read the photometry data
try:
    photometry_df = pd.read_csv(photometry_file)
except FileNotFoundError:
    print(f"Error: {photometry_file} not found.")
    exit()

# Identify the filter flux columns (all columns after 'dec')
flux_columns = photometry_df.columns[3:]

# Find rows where all flux columns are zero
missing_sources_df = photometry_df[(photometry_df[flux_columns] == 0).all(axis=1)]

# Select only the object_id, ra, and dec columns
result_df = missing_sources_df[['object_id', 'ra', 'dec']]

# Save the result to a new CSV file
result_df.to_csv(output_file, index=False)

print(f"Found {len(result_df)} sources with all zero fluxes.")
print(f"Saved to {output_file}")
