"""
salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
srun -n 4 python data_pip.py
"""

import os
import time
import logging
import itertools
from pathlib import Path

import numpy as np

from mockfactory import Catalog, sky_to_cartesian, setup_logging
import lsstypes as types


logger = logging.getLogger('data_pip') 


def select_region(ra, dec, region=None):
    # print('select', region)
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
    if footprint is None: load_footprint()
    north, south, des = footprint.get_imaging_surveys()
    mask_des = des[hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    raise ValueError('unknown region {}'.format(region))


def get_proposal_mattrs(tracer):
    if 'BGS' in tracer:
        mattrs = dict(boxsize=4000., cellsize=10)
    elif 'LRG+ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'LRG' in tracer:
        mattrs = dict(boxsize=7000., cellsize=10)
    elif 'ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'QSO' in tracer:
        mattrs = dict(boxsize=10000., cellsize=10)
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')
    #mattrs.update(cellsize=30)
    return mattrs

def apply_wntmp(ntile, ntmp_table, method='ntmp'):
    frac_missing_pw, frac_zero_prob = ntmp_table
    if method == 'ntmp':
        toret = 1 - frac_missing_pw[ntile]
    elif method == 'ntzp':
        toret = 1 - frac_zero_prob[ntile]
    else:
        raise NotImplementedError(f'unknown method {method}')
    #ref = apply_wntmp_bak(ntile, frac_missing_pw, frac_zero_prob, ntile_range=[0,15], randoms=True)[0]
    #assert np.allclose(toret, ref)
    return toret

# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    # if not np.issubdtype(array.dtype, np.unsignedinteger):
    #     raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def _compute_ntmp(bitweights, loc_assigned, ntile):
    """
    nbits = 64 * np.shape(bitweights)[1]
    recurr = prob_obs * nbits
    """
    # Input: list of bitweights
    nbits = 8 * sum(weight.dtype.itemsize for weight in bitweights)
    recurr = popcount(*bitweights)
    wiip = (nbits + 1) / (recurr + 1)
    zero_prob = (recurr == 0) & (~loc_assigned)
    
    #print(np.sum(zerop_msk))
    sum_ntile = np.bincount(ntile)
    sum_zero_prob = np.bincount(ntile, weights=zero_prob)
    sum_loc_assigned = np.bincount(ntile, weights=loc_assigned)
    sum_wiip = np.bincount(ntile, weights=loc_assigned * wiip)
    mask_zero_ntile = sum_ntile == 0
    frac_zero_prob = np.divide(sum_zero_prob, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    frac_missing_pw = np.divide(sum_ntile - sum_wiip, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    return frac_missing_pw, frac_zero_prob


def compute_ntmp(full_data_fn):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    ntmp = None
    if mpicomm.rank == 0:
        import fitsio
        catalog = fitsio.read(full_data_fn)
        ntmp = _compute_ntmp(_format_bitweights(catalog['BITWEIGHTS']), catalog['LOCATION_ASSIGNED'], catalog['NTILE'])
    return mpicomm.bcast(ntmp, root=0)



def _format_bitweights(bitweights):
    if bitweights.ndim == 2: return list(bitweights.T)
    return [bitweights]


def get_clustering_rdzw(*fns, kind=None, zrange=None, region=None, tracer=None, weight_type='default', ntmp=None, **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_SYS', 'WEIGHT_ZFAIL', 'WEIGHT_COMP', 'WEIGHT_FKP', 'BITWEIGHTS', 'FRAC_TLOBS_TILES', 'NTILE']
            columns = [col for col in columns if col in catalog.columns()]
            catalog = catalog[columns]
            if zrange is not None:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
                catalog = catalog[mask]
            if 'bitwise' in weight_type:
                mask = (catalog['FRAC_TLOBS_TILES'] != 0)
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)
    
    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT']
        bitwise_weights = []
        if 'bitwise' in weight_type:
            if kind == 'data':
                individual_weight = catalog['WEIGHT'] / catalog['WEIGHT_COMP']
                bitwise_weights = _format_bitweights(catalog['BITWEIGHTS'])
            elif kind == 'randoms' and ntmp is not None:
                individual_weight = catalog['WEIGHT'] * apply_wntmp(catalog['NTILE'], ntmp)
        if 'FKP' in weight_type.upper():
            individual_weight *= catalog['WEIGHT_FKP']
        rdzw.append([catalog['RA'], catalog['DEC'], catalog['Z'], individual_weight] + bitwise_weights)
    rdzw = [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(len(rdzw[0]))]
    for i in range(4): rdzw[i] = rdzw[i].astype('f8')
    return rdzw[:3], rdzw[3:]


def get_full_rdw(*fns, kind='parent', zrange=None, region=None, tracer=None, weight_type='default', ntmp=None, **kwargs):

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            columns = ['RA', 'DEC', 'LOCATION_ASSIGNED', 'BITWEIGHTS', 'NTILE', 'WEIGHT_NTILE']
            columns = [col for col in columns if col in catalog.columns()]
            catalog = catalog[columns]
            if 'fibered' in kind:
                mask = catalog['LOCATION_ASSIGNED']
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)
    
    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT_NTILE']
        bitwise_weights = []
        if 'fibered' in kind and 'data' in kind:
            if ntmp is not None:
                individual_weight /= apply_wntmp(catalog['NTILE'], ntmp)
            bitwise_weights = _format_bitweights(catalog['BITWEIGHTS'])
            if 'bitwise' not in weight_type:  # to be updated
                nbits = 8 * sum(weight.dtype.itemsize for weight in bitwise_weights)
                recurr = popcount(*bitwise_weights)
                wiip = (nbits + 1) / (recurr + 1)
                individual_weight *= wiip
                bitwise_weights = []
        rdzw.append([catalog['RA'], catalog['DEC'], individual_weight] + bitwise_weights)
    rdzw = [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(len(rdzw[0]))]
    for i in range(3): rdzw[i] = rdzw[i].astype('f8')
    return rdzw[:2], rdzw[2:]


def get_clustering_positions_weights(*fns, **kwargs):
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    [ra, dec, z], weights = get_clustering_rdzw(*fns, **kwargs)
    dist = fiducial.comoving_radial_distance(z)
    positions = sky_to_cartesian(dist, ra, dec)
    return positions, weights


def compute_angular_upweights(output_fn, get_data, get_randoms, tracer='ELG'):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, count2, setup_logging
    from lsstypes import ObservableLeaf, ObservableTree
    from lsstypes.types import Count2, Count2Correlation

    fibered_data = Particles(*get_data('fibered_data'), positions_type='rd', exchange=True)
    parent_data = Particles(*get_data('parent_data'), positions_type='rd', exchange=True)

    theta = 10**np.arange(-5, -1 + 0.1, 0.1)
    battrs = BinAttrs(theta=theta)
    wattrs = WeightAttrs(bitwise=dict(weights=fibered_data.get('bitwise_weight')))
    fibered_data_iip = fibered_data.clone(weights=wattrs(fibered_data))  # compute IIP weights

    def get_counts(*particles):
        #setup_logging('error')
        autocorr = len(particles) == 1
        weight = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs)['weight']
        if autocorr:
            norm = wattrs(particles[0]).sum()**2 - wattrs(*(particles * 2)).sum()
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
        # No need to remove auto-pairs, as edges[0] > 0
        return weight / norm
        #return Count2(counts=weight, norm=norm, theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])

    DDfibered = get_counts(fibered_data)
    wattrs = WeightAttrs()
    DDparent = get_counts(parent_data)

    #parent_randoms = Particles(*get_randoms('parent_randoms'), positions_type='rd', exchange=True)
    #DRparent = get_counts(parent_data, parent_randoms)
    #DRfibered = get_counts(fibered_data_iip, parent_randoms)
    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    auw['DD'] = ObservableLeaf(value=np.where(DDfibered == 0., 1., DDparent / DDfibered), **kw)
    #auw['DR'] = ObservableLeaf(value=np.where(DRfibered == 0., 1., DRparent / DRfibered), **kw)
    auw = ObservableTree(list(auw.values()), pairs=list(auw.keys()))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        auw.write(output_fn)
    return auw


def compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_randoms, auw=None, cut=None, ells=(0, 2, 4), edges=None, los='firstpoint', **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum, BinParticle2CorrelationPoles, BinParticle2SpectrumPoles, compute_particle2, compute_particle2_shotnoise, MeshAttrs)

    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)

    data = list(data)
    bitwise_weights = None
    if len(data[1]) > 1:
        bitwise_weights = list(data[1])
        from cucount.jax import BitwiseWeight
        from cucount.numpy import reformat_bitarrays
        data[1] = individual_weight = bitwise_weights[0] * BitwiseWeight(weights=bitwise_weights[1:], p_correction_nbits=False)(bitwise_weights[1:])  # individual weight * IIP weight
    else:  # no bitwise_weights
        data[1] = individual_weight = data[1][0]
    #print(bitwise_weights[0][:10], data[1][:10], data[1].sum(), randoms[1][0].sum())
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(randoms[0], randoms[1][0], attrs=mattrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=10)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    wsum_data1 = data.sum()
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    jax.block_until_ready(spectrum)
    t0 = time.time()
    if cut is not None:
        sattrs = {'theta': (0., 0.05)}
        bin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
        #bin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
        from jaxpower.particle2 import convert_particles
        particles = convert_particles(fkp.particles)
        close = compute_particle2(particles, bin=bin, los=los)
        close = close.clone(num_shotnoise=compute_particle2_shotnoise(particles, bin=bin))
        spectrum = spectrum.clone(value=spectrum.value() - close.clone(norm=norm).value())
    elif bitwise_weights is not None or auw is not None:
        from cucount.jax import WeightAttrs
        from jaxpower.particle2 import convert_particles
        sattrs = {'theta': (0., 0.1)}
        if bitwise_weights is not None:
            data = convert_particles(fkp.data, weights=bitwise_weights + [individual_weight])
        else:
            data = convert_particles(fkp.data, weights=[individual_weight] * 2, index_value=dict(individual_weight=1, negative_weight=1))
        wattrs = WeightAttrs(bitwise=dict(weights=data.get('bitwise_weight')) if bitwise_weights else None,
                             angular=dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value()) if auw is not None else None)
        bin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
        #bin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, wattrs=wattrs, ells=ells)
        DD = compute_particle2(data, bin=bin, los=los)
        DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(data, bin=bin))
        #DD = DD.to_spectrum(spectrum)
        add = DD.clone(norm=norm).value()
        if False: #auw is not None and 'DR' in auw.pairs:
            # I think we can safely ignore the DR term
            data = convert_particles(fkp.data)
            randoms = convert_particles(fkp.randoms, weights=fkp.data.sum() / fkp.randoms.sum() * fkp.randoms.weights)
            wattrs = WeightAttrs(angular=dict(sep=auw.get('DR').coords('theta'), weight=auw.get('DR').value()))
            bin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, sattrs=sattrs, wattrs=wattrs, ells=ells)
            DR = compute_particle2(data, randoms, bin=bin, los=los)
            RD = compute_particle2(randoms, data, bin=bin, los=los)
            add += - DR.clone(norm=norm).value() - RD.clone(norm=norm).value()
        spectrum = spectrum.clone(value=spectrum.value() + add)
    jax.block_until_ready(spectrum)
    logger.info(f'Direct calculation in {time.time() - t0:.2f} s')
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    return spectrum


def compute_fkp_effective_redshift(fkp, cellsize=10., order=2):
    from jax import numpy as jnp
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    from cosmoprimo.utils import DistanceToRedshift
    from jaxpower import compute_fkp2_normalization, compute_fkp3_normalization, FKPField
    fiducial = TabulatedDESI()
    d2z = DistanceToRedshift(lambda z: jnp.array(fiducial.comoving_radial_distance(z)))

    compute_fkp_normalization = {2: compute_fkp2_normalization, 3: compute_fkp3_normalization}[order]

    def compute_z(positions):
        return d2z(jnp.sqrt(jnp.sum(positions**2, axis=-1)))

    if isinstance(fkp, FKPField):
        norm = compute_fkp_normalization(fkp, cellsize=cellsize)
        fkp = fkp.clone(data=fkp.data.clone(weights=data.weights * compute_z(fkp.data.positions)), randoms=randoms.clone(weights=fkp.randoms.weights  * compute_z(fkp.randoms.positions)))
        znorm = compute_fkp_normalization(fkp, cellsize=cellsize)
    else:  # fkp is randoms
        norm = compute_fkp_normalization(fkp, cellsize=cellsize, split=42)
        fkp = fkp.clone(weights=fkp.weights * compute_z(fkp.positions))
        znorm = compute_fkp_normalization(fkp, cellsize=cellsize, split=42)
    return znorm / norm


def compute_jaxpower_window_mesh2_spectrum(output_fn, get_randoms, get_data=None, get_spectrum=None, kind='smooth', **kwargs):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, compute_fkp2_shotnoise, compute_smooth2_spectrum_window, MeshAttrs, get_smooth2_window_bin_attrs, interpolate_window_function, compute_mesh2_spectrum, split_particles)
    spectrum = get_spectrum()
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, norm, edges = spectrum.ells, pole.values('norm')[0], pole.edges('k')
    bin = BinMesh2SpectrumPoles(mattrs, **(dict(edges=edges, ells=ells) | kwargs))
    step = bin.edges[-1, 1] - bin.edges[-1, 0]
    edgesin = np.arange(0., 1.2 * bin.edges.max(), step)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    ellsin = [0, 2, 4]
    output_fn = str(output_fn)

    randoms = get_randoms()
    randoms = ParticleField(randoms[0], randoms[1][0], attrs=mattrs, exchange=True, backend='jax')
    zeff = compute_fkp_effective_redshift(randoms, order=2)
    #if get_data is not None:
    #    from jaxpower import FKPField
    #    data = ParticleField(*get_data(), attrs=mattrs, exchange=True, backend='jax')
    #    zeff = compute_fkp_effective_redshift(FKPField(data=data, randoms=randoms))
    #randoms = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms

    kind = 'smooth'

    if kind == 'smooth':
        correlations = []
        kw = get_smooth2_window_bin_attrs(ells, ellsin)
        compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0, 1])
        # Window computed in configuration space, summing Bessel over the Fourier-space mesh
        coords = jnp.logspace(-3, 5, 4 * 1024)
        for scale in [1, 4]:
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize) #, meshsize=800)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            meshes = []
            for _ in split_particles(randoms.clone(attrs=mattrs2, exchange=True, backend='jax'), None, seed=42):
                alpha = spectrum.attrs['wsum_data1'] / _.sum()
                meshes.append(alpha * _.paint(**kw_paint, out='real'))
            sbin = BinMesh2CorrelationPoles(mattrs2, edges=np.arange(0., mattrs2.boxsize.min() / 2., mattrs2.cellsize.min()), **kw, basis='bessel') #, kcut=(0., mattrs2.knyq.min()))
            #num_shotnoise = compute_fkp2_shotnoise(randoms, bin=sbin)
            correlation = compute_mesh2_correlation(*meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells)) #, num_shotnoise=num_shotnoise)
            del meshes
            if False and jax.process_index() == 0:
                correlation_fn = output_fn.replace('window_mesh2_spectrum', f'window_correlation{scale:d}_bessel_mesh2_spectrum')
                logger.info(f'Writing to {correlation_fn}')
                correlation.write(correlation_fn)
            correlation = interpolate_window_function(correlation, coords=coords, order=3)
            correlations.append(correlation)
        limits = [0, 0.4 * mattrs.boxsize.min(), 2. * mattrs.boxsize.max()]
        weights = [jnp.maximum((coords >= limits[i]) & (coords < limits[i + 1]), 1e-10) for i in range(len(limits) - 1)]
        correlation = correlations[0].sum(correlations, weights=weights)
        flags = ('fftlog',)
        if False and output_fn is not None and jax.process_index() == 0:
            correlation_fn = output_fn.replace('window_mesh2_spectrum', 'window_correlation_bessel_mesh2_spectrum')
            logger.info(f'Writing to {correlation_fn}')
            correlation.write(correlation_fn)
        window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=flags)
    else:
        mesh = randoms.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        window = compute_mesh2_spectrum_window(mesh, edgesin=edgesin, ellsin=ellsin, los=los, bin=bin, pbar=True, flags=('infinite',), norm=norm)
    # Save norm and shotnoise here
    num_shotnoise = next(iter(spectrum)).values('num_shotnoise')[0]
    # Set shotnoise and norm of input spectrum
    observable = window.observable.map(lambda pole, label: pole.clone(value=0. * pole.value(), num_shotnoise=num_shotnoise * (label['ells'] == 0) * np.ones_like(pole.values('num_shotnoise')), norm=norm * np.ones_like(pole.values('norm'))), input_label=True)
    window = window.clone(observable=observable)
    window.attrs.update(spectrum.attrs)
    for pole in window.theory: pole._meta['z'] = zeff
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        window.write(output_fn)
    return window


def compute_theory_for_covariance_mesh2_spectrum(output_fn, spectrum_fns, window_fn, klim=(0., 0.3)):
    import lsstypes as types
    from jaxpower import (ParticleField, MeshAttrs, compute_spectrum2_covariance)
    mean = types.mean([types.read(fn) for fn in spectrum_fns])
    window = types.read(window_fn)

    mattrs = MeshAttrs(**{name: mean.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    covariance = compute_spectrum2_covariance(mattrs, mean)

    sl = slice(0, None, 5)  # rebin to dk = 0.001 h/Mpc
    oklim = (0., 0.35)  # fitted k-range, no need to go to higher k
    smooth = mean.map(lambda pole: pole.clone(k=pole.coords('k', center='mid_if_edges'))).select(k=klim)
    mean = mean.select(k=sl).select(k=oklim)
    window = window.at.observable.select(k=sl).at.observable.select(k=oklim).at.theory.select(k=(0., 1.1 * oklim[1]))
    covariance = covariance.at.observable.select(k=sl).at.observable.select(k=oklim)

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    template = FixedPowerSpectrumTemplate(fiducial='DESI', z=window.theory.get(ells=0).z)
    theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(data=mean.value(concatenate=True), wmatrix=window.value(), ells=mean.ells, k=[pole.coords('k') for pole in mean], kin=window.theory.get(ells=0).coords('k'), ellsin=window.theory.ells, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance.value())

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    theory.init.update(k=smooth.get(0).coords('k'))
    poles = theory(**profiles.bestfit.choice(index='argmax', input=True))
    smooth = smooth.clone(value=poles.ravel())
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        smooth.write(output_fn)
    return smooth


def compute_jaxpower_covariance_mesh2_spectrum(output_fn, get_data, get_randoms, get_theory, get_spectrum):
    import jax
    from jaxpower import (ParticleField, get_mesh_attrs, MeshAttrs, compute_fkp2_covariance_window, compute_spectrum2_covariance, interpolate_window_function, read)
    theory = get_theory()
    spectrum = get_spectrum()
    data, randoms = get_data(), get_randoms()
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    data = ParticleField(data[0], data[1][0], attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(randoms[0], randoms[1][0], attrs=mattrs, exchange=True, backend='jax')
    fftlog = False
    kw = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel') if fftlog else dict(edges={})
    windows = compute_fkp2_covariance_window(randoms, alpha=data.sum() / randoms.sum(),
                                             interlacing=3, resampler='tsc', los='local', **kw)
    if False and output_fn is not None and jax.process_index() == 0:
        for name, window in zip(['WW', 'WS', 'SS'], windows):
            fn = Path(output_fn)
            fn = fn.parent / f'{name}_{fn.name}'
            logger.info(f'Writing to {fn}')
            window.write(fn)

    if fftlog:
        coords = np.logspace(-2, 8, 8 * 1024)
        windows = [interpolate_window_function(window, coords=coords) for window in windows]

    # delta is the maximum abs(k1 - k2) where the covariance will be computed (to speed up calculation)
    covs_analytical = compute_spectrum2_covariance(windows, get_theory(), flags=['smooth'] + (['fftlog'] if fftlog else []), delta=0.4)

    # Sum all contributions (WW, WS, SS), with W = standard window (multiplying delta), S = shotnoise
    # Here we assumed randoms have a negligible contribution to the shot noise in the measurements
    cov = covs_analytical[0].clone(value=sum(cov.value() for cov in covs_analytical))
    cov = cov.at.observable.match(spectrum)
    cov = cov.clone(observable=spectrum)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        cov.write(output_fn)
    return cov


def combine_regions(output_fn, fns):
    combined = types.sum([types.read(fn) for fn in fns])  # for the covariance matrix, assumes observables are independent
    if output_fn is not None:
        logger.info(f'Writing to {output_fn}')
        combined.write(output_fn)
    return combined


def get_catalog_fn(version='dr1-v1.5', kind='data', tracer='LRG', weight_type='bitwise', zrange=(0.8, 1.1), region='NGC', nran=18, **kwargs):
    desi_dir = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/')
    nran_full = 1
    if version == 'dr1-v1.5':
        base_dir = desi_dir / f'Y1/LSS/iron/LSScats'
        if 'bitwise' in weight_type:
            data_dir = base_dir / 'v1.5pip'
        else:
            data_dir = base_dir / 'v1.5'
        if kind == 'data':
            return data_dir / f'{tracer}_{region}_clustering.dat.fits'
        if kind == 'randoms':
            return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(nran)]
        if kind == 'full_data':
            return data_dir / f'{tracer}_full_HPmapcut.dat.fits'
        if kind == 'full_randoms':
            return [data_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.fits' for iran in range(nran_full)]
    elif version == 'dr2-v2':
        base_dir = desi_dir / f'DA2/LSS/loa-v1/LSScats/v2'
        if 'bitwise' in weight_type:
            data_dir = base_dir / 'PIP'
        else:
            data_dir = base_dir / 'nonKP'
        if kind == 'data':
            return data_dir / f'{tracer}_{region}_clustering.dat.fits'
        if kind == 'randoms':
            return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(nran)]
        if kind == 'full_data':
            return base_dir / f'{tracer}_full_HPmapcut.dat.fits'
        if kind == 'full_randoms':
            return [base_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.fits' for iran in range(nran_full)]
    raise ValueError('issue with input args')


def get_measurement_fn(kind='mesh2_spectrum_poles', version='dr2-v1.5', recon=None, tracer='LRG', region='NGC', zrange=(0.8, 1.1), cut=None, auw=None, nran = 18, weight_type='default', **kwargs):
    # base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/')
    # base_dir = base_dir / (f'blinded_{recon}' if recon else 'blinded')
    # base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/blinded_data/{version}/data_splits')
    base_dir = Path(f'/pscratch/sd/s/shengyu/Y3/blinded/{version}/data_splits')
    if cut: cut = '_thetacut'
    else: cut = ''
    if auw: auw = '_auw'
    else: auw = ''
    return str(base_dir / f'{kind}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{weight_type}{auw}{cut}_nran{nran}_o.h5')


if __name__ == '__main__':
    # tracers = [('BGS', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG', (0.8, 1.1)), ('ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1))]
    tracers = [('QSO', (0.8, 2.1))]
    regions = ['NGC', 'SGC']
    versions = ['dr1-v1.5', 'dr2-v2'][:1]
    weight_types = ['default', 'default_fkp', 'default_thetacut', 'default_auw', 'bitwise', 'bitwise_auw'][1:2]
    todo = []
    #todo += ['auw']
    #todo += ['mesh2_spectrum']
    #todo += ['window_mesh2_spectrum']
    #todo += ['covariance_mesh2_spectrum']
    todo += ['mesh2_spectrum']

    setup_logging()

    with_jax = any(td in ['auw', 'mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'] for td in todo)

    if with_jax:
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
        jax.distributed.initialize()
        from jaxpower.mesh import create_sharding_mesh
    else:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
    
    for (tracer, zrange), region, version, weight_type in itertools.product(tracers, regions, versions, weight_types):
        if 'BGS' in tracer:
            tracer = 'BGS_BRIGHT-21.5' if 'dr1' in version else 'BGS_BRIGHT-21.35'
        catalog_args = dict(version=version, region=region, tracer=tracer, zrange=zrange, weight_type=weight_type)
        spectrum_args = dict(**get_proposal_mattrs(catalog_args['tracer']), ells=(0, 2, 4), edges=dict(step=0.001))
        if weight_type.endswith('_thetacut'):
            catalog_args['weight_type'] = weight_type[:-len('_thetacut')]
            spectrum_args['cut'] = 'theta'
        with_auw = weight_type.endswith('_auw')
        if with_auw:
            catalog_args['weight_type'] = weight_type[:-len('_auw')]

        data_fn = get_catalog_fn(kind='data', **catalog_args)
        all_randoms_fn = get_catalog_fn(kind='randoms', **catalog_args)

        if 'bitwise' in catalog_args['weight_type']:
            catalog_args['ntmp'] = compute_ntmp(get_catalog_fn(kind='full_data', **catalog_args))

        get_data = lambda: get_clustering_positions_weights(data_fn, kind='data', **catalog_args)
        get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, kind='randoms', **catalog_args)

        if 'auw' in todo:
            full_data_fn = get_catalog_fn(kind='full_data', **catalog_args)
            all_full_randoms_fn = get_catalog_fn(kind='full_randoms', **catalog_args)
            get_full_data = lambda kind: get_full_rdw(full_data_fn, kind=kind, **catalog_args)
            get_full_randoms = lambda kind: get_full_rdw(*all_full_randoms_fn, kind=kind, **catalog_args)

            output_fn = get_measurement_fn(**catalog_args, kind='angular_upweights')
            with create_sharding_mesh() as sharding_mesh:
                compute_angular_upweights(output_fn, get_full_data, get_full_randoms)

        if with_auw:
            jax.experimental.multihost_utils.sync_global_devices('auw')
            spectrum_args['auw'] = types.read(get_measurement_fn(**catalog_args, kind='angular_upweights'))

        if 'combine' in todo and region == regions[0]:
            for kind in ['mesh2_spectrum_poles', 'window_mesh2_spectrum_poles', 'covariance_mesh2_spectrum_poles'][1:]:
                kw = dict(kind=kind, **catalog_args, **spectrum_args)
                fns = [get_measurement_fn(**(kw | dict(region=region))) for region in regions]
                output_fn = get_measurement_fn(**(kw | dict(region='GCcomb')))
                combine_regions(output_fn, fns)

        if 'mesh2_spectrum' in todo:
            output_fn = get_measurement_fn(**catalog_args, **spectrum_args, kind='mesh2_spectrum_poles')
            spectrum_args2 = dict(spectrum_args)
            if 'dr2' in version:
                spectrum_args2.update(ells=[0], edges={'step': 0.02})
            with create_sharding_mesh() as sharding_mesh:
                compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_randoms, **spectrum_args2)

        if 'window_mesh2_spectrum' in todo:
            output_fn = get_measurement_fn(**catalog_args, kind='window_mesh2_spectrum_poles')
            with create_sharding_mesh() as sharding_mesh:
                get_spectrum = lambda: compute_jaxpower_mesh2_spectrum(None, get_data, get_randoms, **spectrum_args)
                compute_jaxpower_window_mesh2_spectrum(output_fn, get_randoms, get_data=get_data, get_spectrum=get_spectrum)

        if 'covariance_mesh2_spectrum' in todo:
            jax.experimental.multihost_utils.sync_global_devices('covariance')

            def _get_theory():
                window_fn = get_measurement_fn(**catalog_args, kind='window_mesh2_spectrum_poles')
                window = types.read(window_fn)
                kmax = window.observable.get(0).edges('k').max()
                from compare_cubic import get_measurement_fn as get_cubic_measurement_fn
                from compare_cubic import get_zsnap_from_z
                cubic_tracer = {'BGS': 'BGS-21.35', 'ELG': 'ELG_LOP'}.get(tracer[:3], tracer[:3])
                version = 'abacus-hf-v2'
                zsnap = get_zsnap_from_z(cubic_tracer, zrange, version=version)
                flavor = {'ELG': 'base_conf_nfwexp'}.get(tracer[:3], 'base')
                cubic_catalog_args = {'version': version, 'tracer': cubic_tracer, 'zsnap': zsnap, 'flavor': flavor, 'los': 'z'}
                spectrum_fns = get_cubic_measurement_fn(imock=None, **cubic_catalog_args, kind='mesh2_spectrum_poles')
                window_fn = get_cubic_measurement_fn(**cubic_catalog_args, kind='window_mesh2_spectrum_poles')
                return compute_theory_for_covariance_mesh2_spectrum(None, spectrum_fns, window_fn, klim=(0., kmax))

            theory = _get_theory()

            def get_theory():
                return theory

            def get_spectrum():
                window_fn = get_measurement_fn(**catalog_args, kind='window_mesh2_spectrum_poles')
                window = types.read(window_fn)
                window = window.at.theory.match(theory)
                spectrum = window.dot(theory, return_type=None, zpt=False)  # convolved theory power spectrum, inherits shotnoise and norm from window.observable
                spectrum.attrs.update(window.attrs)
                spectrum = spectrum.select(k=(0., theory.get(0).edges('k').max()))
                return spectrum

            output_fn = get_measurement_fn(**catalog_args, kind='covariance_mesh2_spectrum_poles')
            with create_sharding_mesh() as sharding_mesh:
                compute_jaxpower_covariance_mesh2_spectrum(output_fn, get_data, get_randoms, get_theory=get_theory, get_spectrum=get_spectrum)
                jax.clear_caches()


    if with_jax:
        jax.distributed.shutdown()