
import io
from astropy.table import Table

def format_candidate(key: str, file_path: str = 'candidates.list'):
    """
    Parses the candidates.list file and returns the subtable for a given key.

    Parameters
    ----------
    key : str
        The key to search for in the file (e.g., 'MACS0152-28').
    file_path : str, optional
        The path to the candidates.list file, by default 'candidates.list'.

    Returns
    -------
    tuple
        A tuple containing the Astropy Table and the redshift.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    in_block = False
    block_lines = []
    redshift = None

    for line in lines:
        if line.strip().startswith(key):
            in_block = True
            redshift = float(line.strip().split('z=')[1])
            continue

        if in_block:
            if line.strip() == '':
                break
            block_lines.append(line)

    if not block_lines:
        raise ValueError(f"Key '{key}' not found in {file_path}")

    # The data seems to be fixed-width, but also space-separated.
    # Using astropy's ASCII reader should be robust enough.
    # The columns are: RA, Dec, ID, and an optional description.
    # We'll use a space delimiter and handle the description column carefully.
    
    # Pre-process lines to handle the description which can contain spaces
    data_for_table = []
    for line in block_lines:
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 3:
            continue # Skip empty or malformed lines
        
        ra, dec, id_val = parts[:3]
        description = parts[3] if len(parts) > 3 else ''
        data_for_table.append(f"{ra} {dec} {id_val} '{description}'")


    table_string = '\n'.join(data_for_table)
    
    try:
        table = Table.read(table_string, format='ascii', names=['RA', 'Dec', 'ID', 'Description'])
    except Exception as e:
        # Fallback for lines that might not parse correctly
        print(f"Error parsing table for key {key}: {e}")
        # Try to read just the first 3 columns if there's an issue
        table_string_3_col = '\n'.join([' '.join(line.strip().split()[:3]) for line in block_lines])
        table = Table.read(table_string_3_col, format='ascii', names=['RA', 'Dec', 'ID'])


    return table, redshift

if __name__ == '__main__':
    # Example usage:
    key_to_find = 'MACS0152-28'
    try:
        astropy_table, z = format_candidate(key_to_find)
        print(f"Redshift for {key_to_find}: {z}")
        print(astropy_table)

        key_to_find = 'MACS0018+16'
        astropy_table, z = format_candidate(key_to_find)
        print(f"Redshift for {key_to_find}: {z}")
        print(astropy_table)

    except ValueError as e:
        print(e)

