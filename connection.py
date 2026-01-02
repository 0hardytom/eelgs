import numpy as np
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
# This script assumes that other necessary modules (like pf, lr, ift) 
# and data variables (like tab, SNR_mask, EW_mask) are already loaded.

fig, ax = pf.create_plot()
ax2 = fig.add_axes((1.025,0,0.1,1))
specax = fig.add_axes((0,1.05,1,0.2))
specax.set_ylabel(r'Flux [Norm]', fontsize=12)
specax.set_xlabel(r'Rest Wavelength, $\lambda_0$ [$\AA$]')
specax.xaxis.tick_top()
specax.xaxis.set_label_position('top') 
specax.set_xlim(4800,5100)
specax.set_xticklabels(specax.get_xticklabels(),fontsize=12)

ax.set_xlabel(r'$\log$([N II]/H$\alpha$)')
ax.set_ylabel(r'$\log$([O III]/H$\beta$)')

xvals = np.log10(tab['nii6583_flux']/tab['halpha_flux'])
yvals = np.log10((tab['oiii5007_flux']+tab['oiii4959_flux'])/tab['hbeta_flux'])

xerr = np.abs(xvals*np.sqrt( (tab['nii6583_flux_err']/tab['nii6583_flux'])**2 + (tab['halpha_flux_err']/tab['halpha_flux'])**2 ))
yerr = np.abs(yvals*np.sqrt( (tab['oiii5007_flux_err']/tab['oiii5007_flux'])**2 + (tab['hbeta_flux_err']/tab['hbeta_flux'])**2 ))

peamask = (SNR_mask&EW_mask)

classf = lr.classify_bpt(xvals,yvals)
markers = ['o','x','^']
colors = ['#ff004f', "#77aca2", "#0359c3"]

text_coords = -1.5,0

transition = []
transition_pea_coords = None # ADDED: To store coordinates of the transition pea

for i,key in enumerate(list(classf.keys())):
    mask = classf.get(key)

    hbetaDISP = tab['hbeta_vel_disp'][mask]

    ax.errorbar(xvals[mask],yvals[mask], fmt='none', yerr=yerr[mask], ms=4, ecolor='gray', capsize=2)
    sc = ax.scatter(xvals[mask],yvals[mask], marker = markers[i], c=hbetaDISP, s=20, zorder=10, cmap='magma_r', vmax=100, vmin=30)
    fig.colorbar(sc, ax=ax,cax=ax2, label=r'log($V($ H$\beta$ [ms$^{-1}$] )) ')

    current_peamask = mask&peamask
    if key == 'transition':
        transition.append(tab[peamask][0])
        # ADDED: Store the coordinates of the transition pea if it exists
        if np.any(current_peamask):
            transition_pea_coords = (xvals[current_peamask][0], yvals[current_peamask][0])

    ax.scatter(xvals[current_peamask],yvals[current_peamask], s=150, marker='s', edgecolors='#777777', c='None', linewidths=2)

    pea_x_coords = xvals[current_peamask]
    pea_y_coords = yvals[current_peamask]

    for pea_x, pea_y in zip(pea_x_coords, pea_y_coords):
        arrow = FancyArrowPatch(
            (text_coords[0], text_coords[1]),
            (pea_x, pea_y),
            arrowstyle='->,head_length=5,head_width=3',
            connectionstyle="arc3,rad=.2",
            color='k',
            linestyle='-',
            linewidth=2,
            zorder=15,
            shrinkA=10, 
            shrinkB=5  
        )
        ax.add_patch(arrow)
shift = 0.3
ax.text(text_coords[0]-shift,text_coords[1],'Peas', fontsize=15)


ax.set_xlim(-2.1,.6)
ax.set_ylim(-.55,1.5)

grid = np.linspace(-2.5,0, 100)
grid2 = np.linspace(-2.5,.3, 100)

ax.plot(grid,lr.kauffmann03(grid),color='k', ls='--')
ax.plot(grid2,lr.kewley01(grid2),color='k')

ax.text(-1.7,-.3,'Starburst', fontsize=20)
ax.text(0.01,-.3,'AGN', fontsize=20, bbox=dict(facecolor='white', edgecolor='none', zorder=10))
ax2.set_ylim([40,100])

###### PLOT THE SPECTRUM ########

galloc, cube_ift, cluster, spectrum_o, rest_spectrum = ift.get_fromIFU(transition[0])
specax.plot(rest_spectrum.wave.coord(), rest_spectrum.data, color='#ff004f', lw=1, label='Rest Frame Spectrum')

# --- NEW CODE TO CONNECT SPEC AXIS TO SQUARE ---
if transition_pea_coords is not None:
    # This is an estimate of the half-size of the square marker in data coordinates.
    # The marker size `s=150` is in points^2. This conversion is an approximation.
    square_half_side = 0.04

    pea_x, pea_y = transition_pea_coords

    # Define the top corners of the square on the main plot (ax)
    top_left_corner_ax = (pea_x - square_half_side, pea_y + square_half_side)
    top_right_corner_ax = (pea_x + square_half_side, pea_y + square_half_side)

    # Get the limits of the spectrum plot (specax) to connect the lines to its bottom edge
    spec_xmin, spec_xmax = specax.get_xlim()
    spec_ymin = specax.get_ylim()[0]

    # Define the connection points on the bottom of the spectrum plot
    bottom_left_specax = (spec_xmin, spec_ymin)
    bottom_right_specax = (spec_xmax, spec_ymin)

    # Create the connection lines (patches)
    con1 = ConnectionPatch(
        xyA=bottom_left_specax, xyB=top_left_corner_ax,
        coordsA="data", coordsB="data", axesA=specax, axesB=ax,
        color="gray", linestyle="--"
    )
    con2 = ConnectionPatch(
        xyA=bottom_right_specax, xyB=top_right_corner_ax,
        coordsA="data", coordsB="data", axesA=specax, axesB=ax,
        color="gray", linestyle="--"
    )

    # Add the connection lines to the figure
    fig.add_artist(con1)
    fig.add_artist(con2)
# --- END OF NEW CODE ---

pf.fix_plot([ax,ax2, specax])
fig.savefig(f'figs/bpt_vel_wPeas1.png', dpi=600, bbox_inches='tight')
