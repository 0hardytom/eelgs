
import csv
import re

def parse_candidates(file_path):
    candidates = []
    cluster_name = None
    cluster_redshift = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for cluster line
            match = re.match(r'(\S+)\s+z=([\d\.]+)', line)
            if match:
                cluster_name = match.group(1)
                cluster_redshift = float(match.group(2))
                continue

            # Skip lines that don't look like candidate data
            if not (line[0].isdigit() or line.startswith('-')):
                continue

            parts = line.split(maxsplit=4)
            if len(parts) < 4:
                continue
            
            ra, dec, candidate_id, candidate_type = parts[:4]
            description = parts[4] if len(parts) > 4 else ''


            candidates.append({
                'cluster': cluster_name,
                'cluster_redshift': cluster_redshift,
                'ra': ra,
                'dec': dec,
                'id': candidate_id,
                'type': candidate_type,
                'description': description
            })

    return candidates

def write_csv(candidates, output_path):
    if not candidates:
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
        writer.writeheader()
        writer.writerows(candidates)

if __name__ == '__main__':
    input_file = 'oii_lya_candidates.list'
    output_file = 'oii_lya_candidates.csv'
    
    candidates_data = parse_candidates(input_file)
    write_csv(candidates_data, output_file)
    print(f"Successfully converted {input_file} to {output_file}")
