import numpy as np

emlines_wave  = []
emlines_label = []

with open('emlines_info.dat', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',', 1)
            emlines_wave.append(float(parts[0]))
            emlines_label.append(parts[1].strip())

emlines_wave = np.array(emlines_wave)

rest_lambdas = {
    'oiii5007':  5006.84,
    'oiii4959':  4958.91,
    'oii3726':   3726.03,
    'oii3729':   3728.82,
    'halpha':    6562.80,
    'hbeta':     4861.33,
    'hgamma':    4340.46,
    'hdelta':    4101.73,
    'hepsilon':  3970.08,
    'hzeta':     3889.06,
    'heta':      3835.40,
    'oiii4363':  4363.21,
    'neiii':     3868.75,
    'nii6583':   6583.45,
    'nii6548':   6548.05,
    'sii6716':   6716.44,
    'sii6731':   6730.82,
    'nev3426':   3426.00,
    'fevii3760': 3760.00,
    'heii4686':  4685.68,
    'hei5876':   5875.62,
}

TOLERANCE = 2.0

elines_to_fit = []

for name, wav in rest_lambdas.items():
    diffs = np.abs(emlines_wave - wav)
    idx   = np.argmin(diffs)
    if diffs[idx] < TOLERANCE:
        elines_to_fit.append(emlines_label[idx])
    else:
        print(f'# WARNING: {name} (λ={wav:.2f}) unmatched, closest Δ={diffs[idx]:.2f} Å')

print('elines_to_fit = [')
for label in elines_to_fit:
    print(f'    "{label}",')
print(']')