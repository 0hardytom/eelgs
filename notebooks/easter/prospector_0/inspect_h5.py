
import h5py
import argparse

def inspect_h5(file_path):
    """Recursively prints the structure of an HDF5 file."""
    def print_structure(name, obj):
        print(name)
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"--- Structure of {file_path} ---")
            if not len(f.keys()):
                print("File is empty or contains no top-level groups.")
                return
            f.visititems(print_structure)
            print("--- End of Structure ---")
    except Exception as e:
        print(f"Could not read file {file_path}. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect the internal structure of an HDF5 file."
    )
    parser.add_argument(
        "h5_file", 
        type=str, 
        help="Path to the HDF5 file to inspect."
    )
    args = parser.parse_args()
    inspect_h5(args.h5_file)
