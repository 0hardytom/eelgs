import numpy as np
from astropy.io import fits
import csv

def calculate_pivot_wavelength(l, t):
    """
    Calculate the pivot wavelength for a filter.
    """
    # Note: np.trapz is deprecated, but using for consistency with original script.
    # For newer numpy versions, np.trapezoid would be preferred.
    return np.sqrt(np.trapz(t * l, l) / np.trapz(t / l, l))

def main():
    """
    Main function to extract filter information, calculate pivot wavelengths,
    and save the results to a CSV file.
    """
    output_filename = 'filter_wavelengths.csv'
    try:
        with fits.open('filter_list.fits') as hdul, open(output_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Filter', 'Pivot_Wavelength_A'])

            print(f"Processing filters and writing to {output_filename}...")

            for hdu in hdul[1:]:  # Skip the primary HDU
                if isinstance(hdu, fits.BinTableHDU):
                    filter_name = hdu.name
                    data = hdu.data
                    
                    try:
                        l = data['lambda']
                        t = data['throughput']
                        
                        pivot_wave = calculate_pivot_wavelength(l, t)
                        
                        csv_writer.writerow([filter_name, f"{pivot_wave:.2f}"])
                        
                    except KeyError as e:
                        print(f"Could not find columns 'lambda' and 'throughput' in {filter_name}: {e}")
            
            print("Processing complete.")

    except FileNotFoundError:
        print("Error: filter_list.fits not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()