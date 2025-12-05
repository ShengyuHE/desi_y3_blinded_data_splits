import os
import numpy as np
from astropy.table import Table
from pathlib import Path

##### constant #####
CSPEED = 299792.458
PLANCK_COSMOLOGY = {
    "Omega_m": 0.315191868,
    "H_0": 67.36,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "h": 0.6736,
    "A_s": 2.083e-9,
    "logA": 3.034,
    "n_s": 0.9649,
    "N_ur": 2.0328,
    "N_ncdm": 1.0,
    "omega_ncdm": 0.0006442,
    "w_0": -1,
    "w_a": 0.0
}


##### bins settings #####

TRACER_BINS = [("LRG", (0.4,0.6)), ]


REDSHIFT_BIN_LSS  = dict(BGS = [(0.1, 0.4)],
                       LRG = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 
                       ELG = [(0.8, 1.1), (1.1, 1.6)],
                       QSO = [(0.8, 2.1)])

REDSHIFT_BIN_OVERALL = dict(BGS = (0.1, 0.4),
                       LRG = (0.4, 1.1), 
                       ELG = (0.8, 1.6),
                       QSO = (0.8, 2.1))

       
REDSHIFT_ABACUSHF_v1 = dict(BGS = [0.200],
                         LRG = [0.500, 0.725, 0.950],
                         ELG= [0.950, 1.175, 1.475],
                         QSO = [1.400])

REDSHIFT_CUBICBOX = dict(BGS = [0.200],
                         LRG = [0.500, 0.800, 0.800],
                         ELG= [0.800, 1.100],
                         QSO = [1.100])
                      
# REDSHIFT_ABACUS_Y3 = dict(BGS = None,
#                          LRG = [0.500, 0.725, 0.950],
#                          ELG= [0.950, 1.175, 1.475],
#                          QSO = [0.950, 1.250, 1.550, 1.850])

REDSHIFT_EZMOCKS_Y1 = dict(BGS = [0.200],
                         LRG = [0.500, 0.800, 0.800],
                         ELG= [0.950, 1.100],
                         QSO = [1.100])

NRAN = {'LRG': 8, 'ELG': 10, 'QSO': 4}

Y3_EFFECTIVE_VOLUME = dict(BGS = [3.8], 
                           LRG = [4.9, 7.6, 9.8],
                           ELG = [5.8, 8.3],
                           
                           QSO = [2.7])
Y3_SMOOTHING = {'BGS_ANY-02':[15.], 'LRG': [15.], 'LRG+ELG_LOPnotqso': [15.], 'ELG_LOPnotqso': [15.], 'QSO': [30.]}
Y3_NRAN = {'LRG': 8, 'LRG+ELG_LOPnotqso': 10, 'ELG_LOPnotqso': 10, 'QSO': 4}
Y3_BOXSIZE = {'LRG': 7000., 'LRG+ELG_LOPnotqso': 9000., 'ELG_LOPnotqso': 9000., 'QSO': 10000.}


RSF_COV_ERROR = dict(LRG = [0.0078, 0.0050, 0.0039],
                     ELG = [0.0066, 0.0046],
                     QSO = [0.0141])

RSF_CUBIC_ERROR = dict(BGS = [0.6434],
                       LRG = [0.4990, 0.3217, 0.2495],
                       ELG = [0.4216, 0.2946],
                       QSO = [0.9056])

RSF_EZMOCKS_ERROR = dict(BGS = None,
                         LRG = [0.2105, 0.1357, 0.1053],
                         ELG = [0.1778, 0.1243],
                         QSO = [0.3820])


##### Notations #####
TRACER_CUTSKY_INFO = {
    'LRG': {'tracer_type': 'LRG', 'fit_range': '0p4to1p1'},
    'ELG': {'tracer_type': 'ELG_LOP','fit_range': '0p8to1p6'},
    'QSO': {'tracer_type': 'QSO','fit_range': '0p8to3p5'},
}

##### Functions #####
def GET_RECON_BIAS(tracer='LRG', grid_cosmo=None): # need update for different cosmologies
    if tracer.startswith('BGS'):
        f=  0.682
        bias = {'000': 1.5, '001': 1.7, '002': 1.6, '003': 1.6, '004': 1.8}
        smoothing_radius = 15.
    elif tracer.startswith('LRG+ELG'):
        f = 0.85
        bias = {'000': 1.6, '001': 1.7, '002': 1.6, '003': 1.6, '004': 1.8}
        smoothing_radius = 15.
    elif tracer.startswith('LRG'):
        f =  0.834
        bias = {'000': 2.0, '001': 2.1, '002': 1.9, '003': 1.9, '004': 2.2}
        smoothing_radius = 15.
    elif tracer.startswith('ELG'):
        f= 0.9
        bias = {'000': 1.2, '001': 1.3, '002': 1.2, '003': 1.2, '004': 1.4}
        smoothing_radius = 15.
    elif tracer.startswith('QSO'):
        f= 0.928
        bias = {'000': 2.1, '001': 2.3, '002': 2.1, '003': 2.1, '004': 2.4}
        smoothing_radius = 30.
    else:
        raise ValueError('unknown tracer {}'.format(tracer))
    if grid_cosmo is None:
        bias = bias['000']
    else:
        bias = bias[grid_cosmo]
    return f, bias, smoothing_radius

def SKY_TO_CARTESIAN(rdd, degree=True):
    conversion = 1.
    if degree: conversion = np.pi / 180.
    ra, dec, dist = rdd
    cos_dec = np.cos(dec * conversion)
    x = dist * cos_dec * np.cos(ra * conversion)
    y = dist * cos_dec * np.sin(ra * conversion)
    z = dist * np.sin(dec * conversion)
    return [x, y, z]

def GET_LINE_CONFUSION(tracer):
    # if tracer == 'BGS':
    
    if tracer == 'LRG':
        return 0

    if tracer == 'ELG':
        return 0

    if tracer == 'QSO':
        # line_set = [1215.67, 1549, 1908, 2800, 3727, 3868, 
        #             4101, 4340, 4861, 4958, 5007, 6562]
        # name_set = [r'Ly$\alpha$', 'C[IV]', 'C[III]', 'Mg[II]', 'O[II]', 'Ne[III]',
        #             r'H$\delta$', r'H$\gamma$', r'H$\beta$', 'O[III]1', 'O[III]2', r'H$\alpha$']
        line_set = [2800, 1908, 4340, 1549, 1215.67]
        name_set = ['Mg[II]', 'C[III]', r'H$\gamma$', 'C[IV]',r'Ly$\alpha$',]
    return (line_set, name_set)

def GET_REPEATS_NUMBER(tracer, z1, z2, table_path='/global/homes/s/shengyu/desi_y3_redshift_errors/main/repeat_obs/results/repeat_numbers.csv'):
    """
    Read repeat_numbers.csv and return (N, N_p, N_n)
    for a given tracer and z-range (z1, z2).
    """
    # Load table
    tab = Table.read(table_path, format='ascii.csv')
    # Construct tag
    tag = f"{tracer}_{z1}_{z2}"
    # Find matching row
    mask = tab['tag'] == tag
    if not np.any(mask):
        raise ValueError(f"Tag {tag} not found in table {table_path}")
    row = tab[mask][0]
    return row['N'], row['N_p'], row['N_n']

def GET_REPEATS_DV(tracer, zmin, zmax, inverse = False, repeat_dir = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'):
    d = Table.read(f'{repeat_dir}/{tracer}repeats.fits', hdu=1)
    sel      = np.full(len(d),True)
    sel = np.isfinite(d['Z1']) & np.isfinite(d['Z2'])
    if inverse is True:
        selz = ((zmin<d["Z2"])&(d["Z2"]<zmax))
        d_zbin = d[sel & selz]
        dv_zbin = (d_zbin['Z1']-d_zbin['Z2'])/(1+d_zbin['Z2'])*CSPEED
    else:
        selz = ((zmin<d["Z1"])&(d["Z1"]<zmax))
        # d_zbin = d[sel]
        d_zbin = d[sel & selz]
        dv_zbin = (d_zbin['Z2']-d_zbin['Z1'])/(1+d_zbin['Z1'])*CSPEED
    dv = np.asarray(dv_zbin, float)
    dv = dv[np.isfinite(dv)]
    return dv

def GET_CTHR(tracer):
    if tracer in ['BGS', 'LRG', 'ELG']:
        cthr = 1000
    elif tracer in ['QSO']:
        cthr = 10000
    elif tracer in ['QSO_3cut']:
        cthr = 3000
    return cthr


def get_des_mask(ra, dec, polygon_dir='./', if_deg=True):
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon
    from matplotlib.path import Path
    if polygon_dir is None:
        polygon_dir = os.path.join(os.path.dirname(__file__), 'mask_fp')
    pol = np.load(os.path.join(polygon_dir, 'des_footprint.npy'), allow_pickle=True)
    pol[0] *= -1
    pol[0] = np.remainder(pol[0] + 360, 360)
    pol[0][pol[0] > 180] -= 360
    polygon = Polygon(np.radians(pol).T)
    if if_deg:
        r = np.remainder(ra + 360, 360)
        r[r > 180] -= 360
        r = -r
        ra = np.radians(r)
        dec = np.radians(dec)
    return polygon.contains_points(np.array([ra, dec]).T)

def SELECT_REGION(ra, dec, region=None):
    # print('select', region)
    import numpy as np
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    # if footprint is None: load_footprint()
    # north, south, des = footprint.get_imaging_surveys()
    # mask_des = des[hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return get_des_mask(ra, dec)
    if region == 'SnoDES':
        return mask_s & (~get_des_mask(ra, dec))
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~get_des_mask(ra, dec))
    raise ValueError('unknown region {}'.format(region))


'''
RSF_COV_ERROR = dict(LRG = [0.319, 0.256, 0.226],
                     ELG = [0.294, 0.245],
                     QSO = [0.430])

RSF_CUBIC_ERROR = dict(BGS = [0.725],
                       LRG = [0.639, 0.513, 0.452],
                       ELG = [0.587, 0.491],
                       QSO = [0.861])

RSF_EZMOCKS_ERROR = dict(BGS = None,
                         LRG = [0.553, 0.444, 0.391],
                         ELG = [0.509, 0.425],
                         QSO = [0.745])

NUMBER_Y3 = dict(BGS = 1188526, LRG = 4468483, ELG = 6534844, QSO = 2062839)
NUMBER_CUBICBOX = dict(BGS = 4309368, LRG = 2982225, ELG= 6747351, QSO = 221666)
NUMBER_COVBOX = dict(LRG = 63112, ELG= 105268, QSO = 4004)
RSF_CUBIC_ERROR = dict(BGS= 1.904, LRG = 0.8169, ELG= 1.016, QSO = 0.3278)
RSF_COV_ERROR = dict(LRG = 0.1187, ELG= 0.1271, QSO = 0.0443)
NUMBER_CUBICBOX = dict(LRG = [2982225, 2982225, 2982225],
                         ELG= [6747351, 3367861, 3367861],
                         QSO = [218809, 221666, 221666, 221666])   
RSF_ERROR_CUBICBOX = dict(LRG = [0.817, 0.817, 0.817],
                         ELG= [1.016, 0.718, 0.718],
                         QSO = [0.326, 0.328, 0.328, 0.328])
Y3_REDSHIFT_TRACERS_NUMBER = dict(LRG = [1052151, 1613562, 1802770],
                                  ELG= [2737573, 3797271, 3797271],
                                  QSO = [2062839, 2062839, 2062839, 2062839])
LRG = 1052151+1613562+1802770 = 4468483 -- 2982225
ELG = 2737573 + 3797271 = 6534844  -- 6747351
QSO = 2062839 -- 3367861

'''