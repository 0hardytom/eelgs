

import re
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

def parse_file(filename):
    """
    Parses the oii_lya_candidates.list file and returns an astropy table.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    ra_list = []
    dec_list = []
    wavelength_list = []
    line_list = []
    cluster_list = []
    cluster_z_list = []
    comment_list = []

    current_cluster = None
    current_z = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for cluster header line
        if 'z=' in line:
            parts = line.split('z=')
            current_cluster = parts[0].strip()
            current_z = float(parts[1].strip())
            continue

        # Parse data lines
        parts = re.split(r'\s+', line, 4)
        if len(parts) >= 4:
            ra_str, dec_str, wavelength_str, line_type = parts[:4]
            comment = parts[4] if len(parts) > 4 else ''

            ra_list.append(ra_str)
            dec_list.append(dec_str)
            wavelength_list.append(float(wavelength_str))
            line_list.append(line_type)
            comment_list.append(comment)
            cluster_list.append(current_cluster)
            cluster_z_list.append(current_z)

    # Create astropy table
    data = {
        'ra': ra_list,
        'dec': dec_list,
        'wavelength': wavelength_list,
        'line': line_list,
        'cluster': cluster_list,
        'cluster_z': cluster_z_list,
        'comment': comment_list
    }
    table = Table(data)

    # Convert coordinates to degrees
    coords = SkyCoord(table['ra'], table['dec'], unit=(u.hourangle, u.deg))
    table['ra'] = coords.ra.deg
    table['dec'] = coords.dec.deg

    # Reorder columns
    table = table['ra', 'dec', 'wavelength', 'line', 'cluster', 'cluster_z', 'comment']

    return table

if __name__ == '__main__':
    table = parse_file('oii_lya_candidates.list')
    print(table)
    table.write('oii_lya_candidates.ecsv', overwrite=True)
    print("\nTable saved to oii_lya_candidates.ecsv")


