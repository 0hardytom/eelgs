#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import esutil.coords as coords
import os
from shapely.geometry import Polygon
from shapely.ops import unary_union

def parse_stripes(par_file='sdss_stripe.par'):
    """
    Parses the sdss_stripe.par file and returns a list of stripe definitions.
    Each stripe is a dictionary with eta, lambdaMin, lambdaMax.
    """
    stripes = []
    with open(par_file, 'r') as f:
        for line in f:
            if line.strip().startswith('stripedef'):
                parts = line.strip().split()
                try:
                    # For southern stripes, the columns are different
                    if int(parts[1]) in [76, 82, 86]:
                         stripe_def = {
                            'stripeNumber': int(parts[1]),
                            'dec': float(parts[2]),
                            'raMin': float(parts[3]),
                            'raMax': float(parts[4]),
                        }
                    else:
                        stripe_def = {
                            'stripeNumber': int(parts[1]),
                            'eta': float(parts[2]),
                            'lambdaMin': float(parts[3]),
                            'lambdaMax': float(parts[4]),
                        }
                    stripes.append(stripe_def)
                except (ValueError, IndexError):
                    print(f"Could not parse line: {line.strip()}")
    return stripes

def plot_wrapped(ax, ra, dec, color='blue', linewidth=1, label=None, zorder=10):
    """
    Plots a line on a projection plot, handling wrap-around.
    RA is expected in degrees [0, 360].
    """
    # Detect wrap-around points in RA and split the line
    diffs = np.diff(ra)
    wrap_indices = np.where(np.abs(diffs) > 180)[0] + 1
    
    # Convert coordinates to radians for plotting
    # RA needs to be in [-pi, pi] for mollweide
    ra_rad = np.deg2rad(ra - 180)
    dec_rad = np.deg2rad(dec)
    
    start = 0
    for i, end in enumerate(wrap_indices):
        # Only add label to the first segment to avoid duplicates in legend
        current_label = label if i == 0 else None
        ax.plot(ra_rad[start:end], dec_rad[start:end], '-', color=color, linewidth=linewidth, label=current_label)
        start = end
    
    # Plot the last segment
    current_label = label if start == 0 else None
    ax.plot(ra_rad[start:], dec_rad[start:], '-', color=color, linewidth=linewidth, label=current_label, zorder=zorder)


def plot_sdss_stripes(stripes):
    """
    Plots the SDSS stripes on a Mollweide projection, merging the main survey area.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='mollweide')

    main_survey_polygons = []
    southern_stripes = []

    for stripe in stripes:
        # Separate the main survey from the southern stripes based on key
        if 'eta' in stripe:
            eta = stripe['eta']
            lambda_min = stripe['lambdaMin']
            lambda_max = stripe['lambdaMax']
            eta_top = eta + 1.25
            eta_bottom = eta - 1.25
            
            main_survey_polygons.append(Polygon([
                (lambda_min, eta_bottom),
                (lambda_max, eta_bottom),
                (lambda_max, eta_top),
                (lambda_min, eta_top)
            ]))
        else:
            southern_stripes.append(stripe)

    # Merge the main survey polygons into a single shape
    if main_survey_polygons:
        merged_survey = unary_union(main_survey_polygons)
        boundary_lambda, boundary_eta = merged_survey.exterior.xy
        ra_boundary, dec_boundary = coords.sdss2eq(np.array(boundary_lambda), np.array(boundary_eta))
        plot_wrapped(ax, ra_boundary, dec_boundary, color='red', linewidth=1.5, label='SDSS Main Survey')

    # Plot the individual southern stripes directly in RA/Dec
    for i, stripe in enumerate(southern_stripes):
        dec_center = stripe['dec']
        ra_min = stripe['raMin']
        ra_max = stripe['raMax']
        dec_top = dec_center + 1.25
        dec_bottom = dec_center - 1.25
        ra_poly = np.array([ra_min, ra_max, ra_max, ra_min, ra_min])
        dec_poly = np.array([dec_bottom, dec_bottom, dec_top, dec_top, dec_bottom])
        label = 'SDSS Southern Stripes' if i == 0 else None
        plot_wrapped(ax, ra_poly, dec_poly, color='blue', linewidth=1, label=label)

    # --- Add CANDELS Fields ---
    # GOODS-S Field (approx. 10' x 16')
    ra_center_gs = 53.125
    dec_center_gs = -27.805
    ra_width_gs = 16 / 60. / np.cos(np.deg2rad(dec_center_gs)) # width in RA degrees
    dec_width_gs = 10 / 60. # width in Dec degrees
    ra_gs = np.array([ra_center_gs - ra_width_gs/2, ra_center_gs + ra_width_gs/2, ra_center_gs + ra_width_gs/2, ra_center_gs - ra_width_gs/2, ra_center_gs - ra_width_gs/2])
    dec_gs = np.array([dec_center_gs - dec_width_gs/2, dec_center_gs - dec_width_gs/2, dec_center_gs + dec_width_gs/2, dec_center_gs + dec_width_gs/2, dec_center_gs - dec_width_gs/2])
    plot_wrapped(ax, ra_gs, dec_gs, color='green', linewidth=1.5, label='CANDELS/GOODS-S')

    # COSMOS Field (approx. 1.4 deg x 1.4 deg)
    ra_center_cosmos = 150.117
    dec_center_cosmos = 2.206
    ra_width_cosmos = 1.4 / np.cos(np.deg2rad(dec_center_cosmos))
    dec_width_cosmos = 1.4
    ra_cosmos = np.array([ra_center_cosmos - ra_width_cosmos/2, ra_center_cosmos + ra_width_cosmos/2, ra_center_cosmos + ra_width_cosmos/2, ra_center_cosmos - ra_width_cosmos/2, ra_center_cosmos - ra_width_cosmos/2])
    dec_cosmos = np.array([dec_center_cosmos - dec_width_cosmos/2, dec_center_cosmos - dec_width_cosmos/2, dec_center_cosmos + dec_width_cosmos/2, dec_center_cosmos + dec_width_cosmos/2, dec_center_cosmos - dec_width_cosmos/2])
    plot_wrapped(ax, ra_cosmos, dec_cosmos, color='purple', linewidth=1.5, label='CANDELS/COSMOS')
    
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.grid(True)
    ax.legend(loc='upper right')

    tick_angles = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    tick_labels = ['2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h']
    ax.set_xticks(np.deg2rad(tick_angles))
    ax.set_xticklabels(tick_labels)

    plt.title('SDSS Survey Stripes with CANDELS Fields')
    
    output_filename = 'sdss_candels_fields.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {os.path.abspath(output_filename)}")


if __name__ == '__main__':
    par_file = 'sdss_stripe.par'
    if not os.path.exists(par_file):
        print(f"Error: {par_file} not found in {os.getcwd()}")
    else:
        stripes_data = parse_stripes(par_file)
        plot_sdss_stripes(stripes_data)
