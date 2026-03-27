from astropy.coordinates import SkyCoord
from astropy.io import ascii
import astropy.units as u
from astropy.table import Table
from mpdaf.obj import Cube
import numpy as np

#### FIRST WE NEED TO CREATE A DICTIONARY TO KEY THE CUBE
leadlines = Table(ascii.read('leadlines.csv'))
leadlines.add_index('key')

keys = np.unique(leadlines['key'])
fullnames = []
exlocs = []

for k in keys:
    rows = leadlines.loc[k]
    top = rows[0]
    fullnames.append(top['cluster'])
    exlocs.append(top['dir'])

refcat = Table([keys,fullnames,exlocs], names=('key','fullname','dir'))
refcat.add_index('key')

#### THEN WE DO I/O

cat = Table(ascii.read('temp_bin../allsourcesNOSTACK.csv'))

height = len(cat['ra'])
bands = ["WFC3_F502N", "WFC3_F606W", "WFC3_F625W", "WFC3_F656N","WFC3_F775W"]
columns = ['object_id','ra','dec']+bands
grid = np.zeros([height, len(columns)], dtype=np.float64)

tab = Table(grid, names=columns)
tab['ra'] = cat['ra']
tab['dec'] = cat['dec']
tab['object_id'] = cat['object_id']
tab.add_index('object_id')


#### THEN WE DO EXTRACTION ####

for k in keys:
    #write over new data after each cube load to prevent catastrophic failure
    tab.write('MUSE_photometry.csv', overwrite=True)
    try:
        #now extract sources
        sources_in_cluster = cat[cat['name']==k]
        info = refcat.loc[k]
        loc = '/Volumes/Expansion/exp_thardy/'+info['dir']+'/'+info['fullname']+'_COMBINED_CUBE_MED_FINAL.fits''
        cluster_cube = Cube(loc)

        for ii, id in enumerate(sources_in_cluster['object_id']):
            tab_current = tab.loc[id]
            eelg_location = SkyCoord(ra=tab_current['ra']*u.deg, dec=tab_current['dec']*u.deg)
            cutout = cluster_cube.subcube_circle_aperture((eelg_location.dec,eelg_location.ra), 2*u.arcsec)

            for b in bands:

                band_cutout = cutout.get_band_image(c)
                flux = np.nansum(band_cutout.data)*(1e-20)*(u.erg)/(u.s*u.cm**2) # we have lost the AA dependency as this is now an image
                ##### WE NEED TO NOW CONVERT TO JANSKY.... ###

                ### finally ipdate the tab_current ##




    except Exception as e:
        print('Failed; error - ', e)
        