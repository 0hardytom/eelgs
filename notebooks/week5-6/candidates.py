
import io
from astropy.table import Table

class Candidates:
    def __init__(self, file_path: str = 'candidates.list'):
        self.file_path = file_path
        self._data = {}
        self._keys = []
        self._parse_file()

    def _parse_file(self):
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        current_key = None
        block_lines = []

        for line in lines:
            stripped_line = line.strip()
            if 'z=' in stripped_line and not stripped_line.startswith(('0', '1', '2')):
                if current_key: # Save previous block
                    self._data[current_key]['lines'] = block_lines
                    block_lines = []

                parts = stripped_line.split(' z=')
                current_key = parts[0]
                redshift = float(parts[1])
                self._keys.append(current_key)
                self._data[current_key] = {'redshift': redshift, 'lines': []}
            elif current_key and stripped_line:
                block_lines.append(stripped_line)
        
        if current_key: # Save the last block
            self._data[current_key]['lines'] = block_lines


    def keys(self):
        return self._keys

    def get_candidate(self, key: str):
        if key not in self._data:
            raise ValueError(f"Key '{key}' not found in {self.file_path}")

        info = self._data[key]
        redshift = info['redshift']
        block_lines = info['lines']

        if not block_lines:
            return Table(), redshift

        # Parse data directly into lists
        ras, decs, oiii_ests, descriptions = [], [], [], []
        has_descriptions = False
        for line in block_lines:
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 3:
                continue  # Skip empty or malformed lines

            ras.append(parts[0])
            decs.append(parts[1])
            oiii_ests.append(parts[2])
            
            if len(parts) > 3:
                descriptions.append(parts[3])
                has_descriptions = True
            else:
                descriptions.append('')

        # Build the table directly
        if has_descriptions:
            table = Table({
                'RA': ras,
                'Dec': decs,
                'OIII_est': oiii_ests,
                'Description': descriptions
            })
        else:
            table = Table({
                'RA': ras,
                'Dec': decs,
                'OIII_est': oiii_ests,
            })
        
        if table:
            table['OIII_est'] = table['OIII_est'].astype(int)

        return table, redshift

if __name__ == '__main__':
    # Example usage:
    candidates = Candidates('candidates.list')
    
    print("Available keys:")
    print(candidates.keys())
    
    key_to_find = 'MACS0152-28'
    try:
        astropy_table, z = candidates.get_candidate(key_to_find)
        print(f"\nRedshift for {key_to_find}: {z}")
        print(astropy_table)
    except ValueError as e:
        print(e)

    key_to_find = 'MACS0018+16'
    try:
        astropy_table, z = candidates.get_candidate(key_to_find)
        print(f"\nRedshift for {key_to_find}: {z}")
        print(astropy_table)
    except ValueError as e:
        print(e)
