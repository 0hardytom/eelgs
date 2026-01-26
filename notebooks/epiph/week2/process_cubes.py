#!/usr/bin/env python
import os
import subprocess

# Define the base directories
input_dir = "/Volumes/Expansion/exp_thardy/cubes_bksubtracted/"
segcube_dir = "/Volumes/Expansion/exp_thardy/cubes_snr/"
catalog_dir = "cat/"

# Create output directories if they don't exist
os.makedirs(segcube_dir, exist_ok=True)
os.makedirs(catalog_dir, exist_ok=True)

# Suffix to identify and remove from filenames
suffix = "_COMBINED_CUBE_MED_FINAL_CSUB.fits"

# List all files in the input directory
try:
    files = os.listdir(input_dir)
except FileNotFoundError:
    print(f"Error: Input directory not found at {input_dir}")
    exit(1)

# Loop through each file in the directory
for filename in files:
    if filename.endswith(suffix):
        # Extract the ID from the filename
        file_id = filename[:-len(suffix)]

        # Define the full paths for input and output files
        input_file = os.path.join(input_dir, filename)
        output_catalog = os.path.join(catalog_dir, f"{file_id}_cat.fits")
        output_segcube = os.path.join(segcube_dir, f"{file_id}_SNR.fits")

        # Construct the command
        command = [
            "lsd_cat_search.py",
            "-i", input_file,
            "-S", "1",
            "-N", "2",
            "-c", output_catalog,
            "--tabvalues", "I,ID,X_PEAK_SN,Y_PEAK_SN,Z_PEAK_SN,DETSN_MAX",
            "--overwrite",
            "--segcube", output_segcube
        ]

        # Print the command to be executed
        print(f"--- Processing {file_id} ---")
        print("Executing command:")
        print(" ".join(command))

        # Execute the command
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Stdout:")
            print(result.stdout)
            print("Stderr:")
            print(result.stderr)
            print(f"--- Successfully processed {file_id} ---")
        except FileNotFoundError:
            print("Error: lsd_cat_search.py not found. Make sure it is in your PATH.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_id}:")
            print("Return code:", e.returncode)
            print("Stdout:")
            print(e.stdout)
            print("Stderr:")
            print(e.stderr)
            print(f"--- Failed to process {file_id} ---")

print("\nAll files processed.")
