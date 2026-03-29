from astropy.io import ascii, fits
from astropy.table import Table, vstack, hstack,join
import numpy as np

def main():
    tab1 = Table(ascii.read('catalogues_raw/photometry_results_1-155.csv'))
    tab2 = Table(ascii.read('catalogues_raw/photometry_results_156-216.csv'))
    spitz_phot = vstack([tab1,tab2])

    useless = ['flux_HST_F125W','flux_HST_F160W','flux_Spitzer_M1_24','flux_Spitzer_M2_70',
    'flux_DES_g','flux_DES_r','flux_DES_i','flux_DES_z','flux_DES_Y']

    spitz_phot.remove_columns(useless)

    to_clean = ['flux_HST_F435W','flux_HST_F606W','flux_HST_F814W','flux_Spitzer_I1_3.6','flux_Spitzer_I2_4.5']

    for c in to_clean:
        spitz_phot[c][spitz_phot[c]<0] = np.nan

    muse_phot = Table(ascii.read('catalogues_raw/MUSE_photometry.csv'))

    to_clean = ['WFC3_F502N','WFC3_F606W','WFC3_F625W','WFC3_F656N','WFC3_F775W']

    for c in to_clean:
        muse_phot[c] = 100*muse_phot[c]#fixes power of ten issue from code
        muse_phot[c][muse_phot[c]<0] = np.nan

    all_phot = join(spitz_phot,muse_phot)
    mask = np.isfinite(all_phot['flux_Spitzer_I1_3.6'])&np.isfinite(all_phot['flux_Spitzer_I2_4.5'])
    all_phot_clean = all_phot[mask]
    all_phot_clean.remove_column('key')

    master = Table(ascii.read('catalogues_raw/allsourcesNOSTACK.csv'))

    master_with_phot = join(master, all_phot_clean)

    to_remove = ['angdisp','foreground','cluster_member','lensed','Z_dir','Z_dir_e','Z_j19','Z_j19_e','R23','R23_e','mean_vel_disp','sterr_vel_disp']
    master_with_phot.remove_columns(to_remove)
    master_with_phot.write('MASTER.csv',overwrite=True)
    test_for_sed = master_with_phot[master_with_phot['object_id']=='16d22m51pt19268571s-24d38m24pt38599595s']
    test_for_sed.write('test_for_sed.csv', overwrite=True)


if __name__ == "__main__":
    main()
