from astropy.coordinates import SkyCoord
from astropy.io import ascii
import astropy.units as u
from astropy.table import Table, Row
from mpdaf.obj import Cube
import numpy as np
import astropy.constants as c

def main():
    #### FIRST WE NEED TO CREATE A DICTIONARY TO KEY THE CUBE
    leadlines = Table(ascii.read('leadlines.csv'))
    leadlines.add_index('key')

    keys = np.unique(leadlines['key'])
    fullnames = []
    exlocs = []

    for k in keys:
        rows = leadlines.loc[k]
        if isinstance(rows, Table):
            top = rows[0]
        else:
            top = rows
        fullnames.append(top['cluster'])
        exlocs.append(top['dir'])

    refcat = Table([keys,fullnames,exlocs], names=('key','fullname','dir'))
    refcat.add_index('key')

    refcat.write('refcat.csv', overwrite=True)

    #### THEN WE DO I/O

    cat = Table(ascii.read('catalogues_raw/allsourcesNOSTACK.csv'))
    filter_info = ascii.read('filter_wavelengths.csv')
    filter_wavelengths = {row['Filter']: (row['Pivot_Wavelength_A'], row['Filter_FWHM_A']) for row in filter_info}


    height = len(cat['ra'])
    bands = ["WFC3_F502N", "WFC3_F606W", "WFC3_F625W", "WFC3_F656N","WFC3_F775W"]
    columns = ['object_id','ra','dec','key']+bands
    grid = np.zeros([height, len(columns)], dtype=np.float64)

    tab = Table(grid, names=columns)
    tab['ra'] = cat['ra']
    tab['dec'] = cat['dec']
    tab['object_id'] = cat['object_id']
    tab['key'] = cat['name']
    tab.add_index('object_id')


    #### THEN WE DO EXTRACTION ####

    for k in keys:
        print('processing cluster '+k)
        #write over new data after each cube load to prevent catastrophic failure
        tab.write('catalogues_raw/MUSE_photometry.csv', overwrite=True)
        try:
            #now extract sources
            print('Finding sources in cluster')
            sources_in_cluster = cat[cat['name']==k]
            info = refcat.loc[k]
            loc = '/Volumes/Expansion/exp_thardy/'+info['dir']+'/'+info['key']+'_COMBINED_CUBE_MED_FINAL.fits'
            print('loading: ', loc)
            cluster_cube = Cube(loc)

            for ii, id in enumerate(sources_in_cluster['object_id']):
                print(f'extracting object {ii} of {len(sources_in_cluster['object_id'])}')
                tab_current = tab.loc[id]
                eelg_location = SkyCoord(ra=tab_current['ra']*u.deg, dec=tab_current['dec']*u.deg)
                cutout = cluster_cube.subcube_circle_aperture((eelg_location.dec.deg,eelg_location.ra.deg), 2)

                for b in bands:

                    band_cutout = cutout.get_band_image(b)
                    flux = np.nansum(band_cutout.data)*(1e-20)*(u.erg/(u.s*u.cm**2)) # we have lost the AA dependency as this is now an image
                    
                    # CONVERT TO MICROJANSKY
                    pivot_wave, fwhm = filter_wavelengths[b]
                    pivot_wave *= u.AA
                    fwhm *= u.AA
                    
                    f_lambda = flux / fwhm
                    f_nu = f_lambda * pivot_wave**2 / c.c
                    ujy = f_nu.to(u.uJy)
                    
                    # Update the table
                    tab.loc[id][b] = ujy.value
            print('cluster complete!')

        except Exception as e:
            print('Failed; error - ', e)

if __name__ == "__main__":
    main()

            