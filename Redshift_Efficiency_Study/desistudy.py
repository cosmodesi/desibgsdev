# coding: utf-8
import os
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits

import multiprocessing
nproc = multiprocessing.cpu_count() // 2

from desispec.io.util import write_bintable
from desitarget.cuts import isBGS_bright, isBGS_faint
from desiutil.log import get_logger, DEBUG
log = get_logger()


def get_predefined_sim_dict(simname):
    seed_multiplier = 164
    try:
        int_sim = int(simname[-2:])
        seed = int(int_sim * seed_multiplier)
    except:
        seed = int( np.abs( hash(simname) ) / seed_multiplier )
    fiducial_settings = {
                    'suffix': simname, 'use_mock': False, 
                    'nsim': 2, 'nspec': 4000, 
                    'zmin': 0.1, 'zmax': 0.6,
                    'rmagmin': 17., 'rmagmax': 19.5,
                    'seed': seed
        }
    if simname == 'sim01':
        simulation_parameters = {  }  # sim01 contains all fiducial values
    elif simname == 'sim02':
        simulation_parameters = {  }  # sim02 contains all fiducial values
    elif simname == 'sim03':
        simulation_parameters = { 'nsim': 10, 'nspec': 1000  }
    elif simname == 'sim04':
        simulation_parameters = { 'nsim': 10, 'nspec': 1000  }
    elif simname == 'sim05':
        simulation_parameters = { 'zmax': 0.8, 'nspec': 800, 'rmagmin': 19.5, 'rmagmax': 20.0  }  
    elif simname == 'sim06':
        simulation_parameters = { 'zmax': 0.8, 'nspec': 800, 'rmagmin': 19.5, 'rmagmax': 20.0 }  
    elif simname == 'sim07':
        simulation_parameters = { 'nsim': 10, 'nspec': 200, 'zmax': 0.8, 'rmagmin': 19.5, 'rmagmax': 20.0  }
    elif simname == 'sim08':
        simulation_parameters = { 'nsim': 10, 'nspec': 200, 'zmax': 0.8, 'rmagmin': 19.5, 'rmagmax': 20.0  }
    else:
        simulation_parameters = {  }  # at this point, all sims use the same sim settings
        
    fiducial_settings.update(simulation_parameters)
    return fiducial_settings


def get_predefined_obs_dict(simname):
    ## Can also define ranges for ref obs conds
        # obs_conds['moonfracmin'], obs_conds['moonfracmax']
        # obs_conds['moonsepmin'], obs_conds['moonsepmax']
        # obs_conds['exptimemin'], obs_conds['exptimemax']
    ## or even bring in fiducial values
        #from desisim.simexp import reference_conditions
        #ref_obsconditions = reference_conditions['BGS']
    fiducial_conditions = {
                    'AIRMASS': 1.0,  
                    'SEEING': 1.1,
                    'MOONALT': -60,
                    'MOONSEP': 180
        }
    if simname == 'sim01' or simname == 'sim05':
        specified_conditions = { 'EXPTIME': 300, 'MOONFRAC': 0.0 }  # sim01 is the fiducial values
    if simname == 'sim02' or simname == 'sim06':
        specified_conditions = { 'EXPTIME': 480, 'MOONFRAC': 0.8, 
                                 'MOONALT': 30,  'MOONSEP': 120    }  
    elif simname == 'sim03' or simname == 'sim07':
        specified_conditions = { 'exptimemin': 300, 'exptimemax': 720,
                                 'MOONALT': 30, 
                                 'MOONFRAC': 0.8, 
                                 'MOONSEP': 120                          }
    elif simname == 'sim04' or simname == 'sim08': 
        specified_conditions = { 'EXPTIME': 600,
                                 'moonfracmin': 0.6, 'moonfracmax': 0.98,
                                 'MOONALT': 30, 
                                 'MOONSEP': 120                          }  
    else:
        specified_conditions = { 'EXPTIME': 300, 'MOONFRAC': 0.0 } # defaults to sim01 (fiducial)
        
    fiducial_conditions.update(specified_conditions)
    return fiducial_conditions


def bgs_sim_spectra(sim, ref_obsconditions, simdir, overwrite=False, verbose=False):
    """Generate spectra for a given set of simulation parameters with 
    the option of overwriting files.

    """
    from desisim.scripts.quickspectra import sim_spectra

    rand = np.random.RandomState(sim['seed'])
    BGS_template_maker = BGStemplates(rand=rand, verbose=verbose)

    # Generate the observing conditions table.
    simdata = bgs_write_simdata(sim, ref_obsconditions, simdir, rand, overwrite=overwrite)
    randseeds = rand.randint(0,2**14,len(simdata)).astype(int)
    for exp, expdata in enumerate(simdata):
        randseed = randseeds[exp]
        # Generate the observing conditions dictionary.  
        obs = simdata2obsconditions(expdata)

        # Generate the rest-frame templates.  Currently not writing out the rest-frame 
        # templates but we could.
        flux, wave, meta = bgs_make_templates(sim, rand, BGS_template_maker)
        redshifts = np.asarray(meta['REDSHIFT'])
        truefile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}-true.fits'.format(sim['suffix'], exp))
        if overwrite or not os.path.isfile(truefile):    
            write_templates(truefile, flux, wave, meta,overwrite=True)

        spectrafile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}.fits'.format(sim['suffix'], exp))
        if overwrite or not os.path.isfile(spectrafile):
            sourcetypes = np.array(["bgs" for i in range(sim['nspec'])])
            sim_spectra(wave, flux, 'bgs', spectrafile, redshift=redshifts, obsconditions=obs, 
                        sourcetype= sourcetypes, seed= randseed, expid= exp)
        else:
            print('File {} exists...skipping.'.format(spectrafile))

def bgs_redshifts(sim, simdir, overwrite=False):
    """Fit for the redshifts.

    """
    from redrock.external.desi import rrdesi    

    for ii in range(sim['nsim']):
        zbestfile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}-zbest.fits'.format(sim['suffix'], ii))
        spectrafile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}.fits'.format(sim['suffix'], ii))

        if overwrite or not os.path.isfile(zbestfile):
            rrdesi(options=['--zbest', zbestfile, '--mp', str(nproc), spectrafile])
        else:
            print('File {} exists...skipping.'.format(zbestfile))    

def bgs_gather_results(sim, simdir, overwrite=False):
    """Gather all the pieces so we can make plots.

    """
    from desispec.io.spectra import read_spectra
    import fitsio

    nspec = sim['nspec']
    nall = nspec * sim['nsim']

    resultfile = os.path.join(simdir, sim['suffix'], 'bgs-{}-results.fits'.format(sim['suffix']))
    if not os.path.isfile(resultfile) or overwrite:
        pass
    else:
        log.info('File {} exists...skipping.'.format(resultfile))
        return

    cols = [
        ('EXPTIME', 'f4'),
        ('AIRMASS', 'f4'),
        ('MOONFRAC', 'f4'),
        ('MOONSEP', 'f4'),
        ('MOONALT', 'f4'),
        ('SNR_B', 'f4'),
        ('SNR_R', 'f4'),
        ('SNR_Z', 'f4'),
        ('TARGETID', 'i8'),
        ('TEMPLATEID', 'i4'),
        ('RMAG', 'f4'),
        ('GR', 'f4'),
        ('D4000', 'f4'),
        ('EWHBETA', 'f4'), 
        ('ZTRUE', 'f4'), 
        ('Z', 'f4'), 
        ('ZERR', 'f4'), 
        ('ZWARN', 'f4')]
    result = Table(np.zeros(nall, dtype=cols))

    result['EXPTIME'].unit = 's'
    result['MOONSEP'].unit = 'deg'
    result['MOONALT'].unit = 'deg'

    # Read the simulation parameters data table.
    simdatafile = os.path.join(simdir, sim['suffix'], 'bgs-{}-simdata.fits'.format(sim['suffix']))
    simdata = Table.read(simdatafile)

    for ii, simdata1 in enumerate(simdata):
        # Copy over some data.
        result['EXPTIME'][nspec*ii:nspec*(ii+1)] = simdata1['EXPTIME']
        result['AIRMASS'][nspec*ii:nspec*(ii+1)] = simdata1['AIRMASS']
        result['MOONFRAC'][nspec*ii:nspec*(ii+1)] = simdata1['MOONFRAC']
        result['MOONSEP'][nspec*ii:nspec*(ii+1)] = simdata1['MOONSEP']
        result['MOONALT'][nspec*ii:nspec*(ii+1)] = simdata1['MOONALT']

        # Read the metadata table.
        truefile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}-true.fits'.format(sim['suffix'], ii))
        if os.path.isfile(truefile):
            log.info('Reading {}'.format(truefile))        
            meta = Table.read(truefile)

            #result['TARGETID'][nspec*ib:nspec*(ii+1)] = truth['TARGETID']
            result['TEMPLATEID'][nspec*ii:nspec*(ii+1)] = meta['TEMPLATEID']
            result['RMAG'][nspec*ii:nspec*(ii+1)] = 22.5 - 2.5 * np.log10(meta['FLUX_R'])
            result['GR'][nspec*ii:nspec*(ii+1)] = -2.5 * np.log10(meta['FLUX_G'] / meta['FLUX_R'])
            result['D4000'][nspec*ii:nspec*(ii+1)] = meta['D4000']
            result['EWHBETA'][nspec*ii:nspec*(ii+1)] = meta['EWHBETA']
            result['ZTRUE'][nspec*ii:nspec*(ii+1)] = meta['REDSHIFT']

        # Read the zbest file. 
        zbestfile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}-zbest.fits'.format(sim['suffix'], ii))
        if os.path.isfile(zbestfile):
            log.info('Reading {}'.format(zbestfile))
            #zbest = fitsio.read(zbestfile, 'ZBEST')
            #from astropy.table import Table
            zbest = Table.read(zbestfile,'ZBEST')
            # Assume the tables are row-ordered!
            result['Z'][nspec*ii:nspec*(ii+1)] = zbest['Z']
            result['ZERR'][nspec*ii:nspec*(ii+1)] = zbest['ZERR']
            result['ZWARN'][nspec*ii:nspec*(ii+1)] = zbest['ZWARN']

        # Finally, read the spectra to get the S/N.
        spectrafile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}.fits'.format(sim['suffix'], ii))
        if os.path.isfile(spectrafile):  
            log.info('Reading {}'.format(spectrafile))
            spec = read_spectra(spectrafile)
            for band in ('b','r','z'):
                for iobj in range(nspec):
                    these = np.where((spec.wave[band] > np.mean(spec.wave[band])-50) * 
                                     (spec.wave[band] < np.mean(spec.wave[band])+50) * 
                                     (spec.flux[band][iobj, :] > 0))[0]
                    result['SNR_{}'.format(band.upper())][nspec*ii+iobj] = (
                        np.median( spec.flux[band][iobj, these] * np.sqrt(spec.ivar[band][iobj, these]) ) 
                    )

    log.info('Writing {}'.format(resultfile))
    write_bintable(resultfile, result, extname='RESULTS', clobber=True)

    
    
    

def bgs_write_simdata(sim, obs_conds, simdir, obsrand, overwrite=False):
    """Build and write a metadata table with the simulation inputs.  
    Currently, the only quantities that can be varied are moonfrac, 
    moonsep, and exptime, but more choices can be added as needed.

    """
    from desispec.io.util import makepath
    simdatafile = os.path.join(simdir, sim['suffix'], 'bgs-{}-simdata.fits'.format(sim['suffix']))
    makepath(simdatafile)

    cols = [
        ('SEED', 'S20'),
        ('NSPEC', 'i4'),
        ('EXPTIME', 'f4'),
        ('AIRMASS', 'f4'),
        ('SEEING', 'f4'),
        ('MOONFRAC', 'f4'),
        ('MOONSEP', 'f4'),
        ('MOONALT', 'f4')]
    simdata = Table(np.zeros(sim['nsim'], dtype=cols))

    simdata['EXPTIME'].unit = 's'
    simdata['SEEING'].unit = 'arcsec'
    simdata['MOONSEP'].unit = 'deg'
    simdata['MOONALT'].unit = 'deg'

    simdata['SEED'] = sim['seed']
    simdata['NSPEC'] = sim['nspec']
    simdata['AIRMASS'] = obs_conds['AIRMASS']
    simdata['SEEING'] = obs_conds['SEEING']
    simdata['MOONALT'] = obs_conds['MOONALT']

    if 'moonfracmin' in obs_conds.keys():
        simdata['MOONFRAC'] = obsrand.uniform(obs_conds['moonfracmin'], obs_conds['moonfracmax'], sim['nsim'])
    else:
        simdata['MOONFRAC'] = obs_conds['MOONFRAC']

    if 'moonsepmin' in obs_conds.keys():
        simdata['MOONSEP'] = obsrand.uniform(obs_conds['moonsepmin'], obs_conds['moonsepmax'], sim['nsim'])
    else:
        simdata['MOONSEP'] = obs_conds['MOONSEP']

    if 'exptimemin' in obs_conds.keys():
        simdata['EXPTIME'] = obsrand.uniform(obs_conds['exptimemin'], obs_conds['exptimemax'], sim['nsim'])
    else:
        simdata['EXPTIME'] = obs_conds['EXPTIME']

    if overwrite or not os.path.isfile(simdatafile):     
        print('Writing {}'.format(simdatafile))
        write_bintable(simdatafile, simdata, extname='SIMDATA', clobber=overwrite)

    return simdata

def simdata2obsconditions(simdata):
    obs = dict(AIRMASS=simdata['AIRMASS'], 
               EXPTIME=simdata['EXPTIME'],
               MOONALT=simdata['MOONALT'],
               MOONFRAC=simdata['MOONFRAC'],
               MOONSEP=simdata['MOONSEP'],
               SEEING=simdata['SEEING'])
    return obs

def bgs_make_templates(sim, rand, BGSmaker):
    """Generate the actual templates.  If using the mock data then iterate 
    until we build the desired number of models after applying targeting cuts, 
    otherwise use specified priors on magnitude and redshift.

    """
    redshift = rand.uniform(sim['zmin'], sim['zmax'], size=sim['nspec'])
    rmag = rand.uniform(sim['rmagmin'], sim['rmagmax'], size=sim['nspec'])
    
    flux, wave, meta = BGSmaker.bgs_templates.make_templates(
            nmodel=sim['nspec'], redshift=redshift, mag=rmag, seed=sim['seed'])
    return flux, wave, meta

def write_templates(outfile, flux, wave, meta,overwrite=True):
    import astropy.units as u
    from astropy.io import fits

    hx = fits.HDUList()
    hdu_wave = fits.PrimaryHDU(wave)
    hdu_wave.header['EXTNAME'] = 'WAVE'
    hdu_wave.header['BUNIT'] = 'Angstrom'
    hdu_wave.header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
    hx.append(hdu_wave)    

    fluxunits = 1e-17 * u.erg / (u.s * u.cm**2 * u.Angstrom)
    hdu_flux = fits.ImageHDU(flux)
    hdu_flux.header['EXTNAME'] = 'FLUX'
    hdu_flux.header['BUNIT'] = str(fluxunits)
    hx.append(hdu_flux)

    hdu_meta = fits.table_to_hdu(meta)
    hdu_meta.header['EXTNAME'] = 'METADATA'
    hx.append(hdu_meta)

    print('Writing {}'.format(outfile))
    hx.writeto(outfile, clobber=True)

class BGStree(object):
    """Build a KD Tree."""
    def __init__(self):
        from speclite import filters
        from scipy.spatial import cKDTree as KDTree
        from desisim.io import read_basis_templates

        self.bgs_meta = read_basis_templates(objtype='BGS', onlymeta=True)
        self.bgs_tree = KDTree(self._bgs())

    def _bgs(self):
        """Quantities we care about: redshift (z), M_0.1r, and 0.1(g-r).

        """
        zobj = self.bgs_meta['Z'].data
        mabs = self.bgs_meta['SDSS_UGRIZ_ABSMAG_Z01'].data
        rmabs = mabs[:, 2]
        gr = mabs[:, 1] - mabs[:, 2]

        return np.vstack((zobj, rmabs, gr)).T        
    
    def query(self, objtype, matrix, subtype=''):
        """Return the nearest template number based on the KD Tree.

        Args:
          objtype (str): object type
          matrix (numpy.ndarray): (M,N) array (M=number of properties,
            N=number of objects) in the same format as the corresponding
            function for each object type (e.g., self.bgs).
          subtype (str, optional): subtype (only for white dwarfs)

        Returns:
          dist: distance to nearest template
          indx: index of nearest template
        
        """
        if objtype.upper() == 'BGS':
            dist, indx = self.bgs_tree.query(matrix)
        else:
            log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
            raise ValueError
                
        return dist, indx

class BGStemplates(object):
    """Generate spectra.  

    """
    def __init__(self, wavemin=None, wavemax=None, dw=0.2, 
                 rand=None, verbose=False):

        from desimodel.io import load_throughput

        self.tree = BGStree()

        # Build a default (buffered) wavelength vector.
        if wavemin is None:
            wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None:
            wavemax = load_throughput('z').wavemax + 10.0
            
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        self.rand = rand
        self.verbose = verbose

        # Initialize the templates once:
        from desisim.templates import BGS
        self.bgs_templates = BGS(wave=self.wave)#, normfilter='sdss2010-r') # Need to generalize this!
        self.bgs_templates.normline = None # no emission lines!

    def bgs(self, data, index=None, mockformat='durham_mxxl_hdf5'):
        """Generate spectra for BGS.

        Currently only the MXXL (durham_mxxl_hdf5) mock is supported.  DATA
        needs to have Z, SDSS_absmag_r01, SDSS_01gr, VDISP, and SEED, which are
        assigned in mock.io.read_durham_mxxl_hdf5.  See also BGSKDTree.bgs().

        """
        from desisim.io import empty_metatable

        objtype = 'BGS'
        if index is None:
            index = np.arange(len(data['Z']))
            
        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'durham_mxxl_hdf5':
            alldata = np.vstack((data['Z'][index],
                                 data['SDSS_absmag_r01'][index],
                                 data['SDSS_01gr'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.bgs_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta
   


