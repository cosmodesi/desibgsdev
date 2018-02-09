
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.io import fits


import multiprocessing
nproc = multiprocessing.cpu_count() // 2



from desispec.io.util import write_bintable
from desitarget.cuts import isBGS_bright, isBGS_faint
from desitarget.mock.mockmaker import BGSMaker
from desiutil.log import get_logger, DEBUG
log = get_logger()

import seaborn as sns
sns.set(style='white', font_scale=1.1, palette='deep')


class desistudy:
    def __init__(self):
        pass

    def plot_subset(wave, flux, truth, nplot=16, ncol=4, these=None, 
                    xlim=None, loc='right', targname='', objtype=''):
        """Plot a random sampling of spectra."""

        nspec, npix = flux.shape
        if nspec < nplot:
            nplot = nspec

        nrow = np.ceil(nplot / ncol).astype('int')

        if loc == 'left':
            xtxt, ytxt, ha = 0.05, 0.93, 'left'
        else:
            xtxt, ytxt, ha = 0.93, 0.93, 'right'

        if these is None:
            these = rand.choice(nspec, nplot, replace=False)
            these = np.sort(these)

        ww = (wave > 5500) * (wave < 5550)

        fig, ax = plt.subplots(nrow, ncol, figsize=(2.5*ncol, 2*nrow), sharey=False, sharex=True)
        for thisax, indx in zip(ax.flat, these):
            thisax.plot(wave, flux[indx, :] / np.median(flux[indx, ww]))
            if objtype == 'STAR' or objtype == 'WD':
                thisax.text(xtxt, ytxt, r'$T_{{eff}}$={:.0f} K'.format(truth['TEFF'][indx]), ha=ha,
                     va='top', transform=thisax.transAxes, fontsize=13)
            else:
                thisax.text(xtxt, ytxt, 'z={:.3f}'.format(truth['TRUEZ'][indx]), ha=ha, 
                     va='top', transform=thisax.transAxes, fontsize=13)

            thisax.xaxis.set_major_locator(plt.MaxNLocator(3))
            if xlim:
                thisax.set_xlim(xlim)
        for thisax in ax.flat:
            thisax.yaxis.set_ticks([])
            thisax.margins(0.2)

        fig.suptitle(targname)
        fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.93)

    def qa_zmag(redshift, mag, maglabel='r (AB mag)', faintmag=20.0):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        _ = ax[0].hist(redshift, bins=100)
        ax[0].set_xlabel('Redshift')
        ax[0].set_ylabel('Number of Galaxies')

        _ = ax[1].hist(mag, bins=100)
        ax[1].axvline(x=faintmag, ls='--', color='k')
        ax[1].set_xlabel(maglabel)
        ax[1].set_ylabel('Number of Galaxies')

        ax[2].scatter(redshift, mag, s=3, alpha=0.75)
        ax[2].axhline(y=faintmag, ls='--', color='k')
        ax[2].set_xlabel('Redshift')
        ax[2].set_ylabel(maglabel)

        plt.subplots_adjust(wspace=0.3)

    def qa_radec(ras,decs):
        fig, ax = plt.subplots()
        ax.scatter(ras,decs, s=1)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')

    def qa_snr(res):
        rmag, snr_b, snr_r = res['RMAG'], res['SNR_B'], res['SNR_R']

        fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        ax[0].scatter(rmag, snr_b, s=40, alpha=0.95, edgecolor='k')
        ax[0].set_ylabel(r'S/N [$b$ channel]')
        ax[0].set_yscale('log')
        ax[0].grid()

        ax[1].scatter(rmag, snr_r, s=40, alpha=0.95, edgecolor='k')
        ax[1].set_yscale('log')
        ax[1].set_xlabel(r'$r_{DECaLS}$ (AB mag)')
        ax[1].set_ylabel(r'S/N [$r$ channel]')
        ax[1].grid()

        plt.subplots_adjust(hspace=0.1)

    def qa_efficiency(res, pngfile=None):
        """Redshift efficiency vs S/N, rmag, g-r color, redshift, 
        and D(4000).

        """
        from matplotlib.ticker import ScalarFormatter

        fig, ax = plt.subplots(3, 2, figsize=(10, 12), sharey=True)
        xlabel = (r'$r_{DECaLS}$ (AB mag)', 'True Redshift $z$', r'S/N [$r$ channel]', 
                  r'S/N [$b$ channel]', r'Apparent $g - r$', '$D_{n}(4000)$')
        for thisax, xx, dolog, label in zip(ax.flat, ('RMAG', 'ZTRUE', 'SNR_R', 'SNR_B', 'GR', 'D4000'), 
                                            (0, 0, 0, 0, 0, 0), xlabel):

            mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = gethist(xx, res)
            thisax.errorbar(mm, e1, ee1, fmt='o', label='ZWARN=0')
            thisax.errorbar(mm, e2, ee2, fmt='o', label='ZWARN=0, $dz/(1+z)<0.003$')
            thisax.errorbar(mm, e3, ee3, fmt='o', label='ZWARN=0, $dz/(1+z)\geq 0.003$')
            #thisax.errorbar(mm, e4, ee4, fmt='o', label='ZWARN>0')

            thisax.set_xlabel(label)
            if dolog:
                thisax.set_xscale('log')
                thisax.xaxis.set_major_formatter(ScalarFormatter())

            thisax.axhline(y=1, ls='--', color='k')
            thisax.grid()
            thisax.set_ylim([0, 1.1])

        ax[0][0].set_ylabel('Redshift Efficiency')
        ax[1][0].set_ylabel('Redshift Efficiency')
        ax[2][0].set_ylabel('Redshift Efficiency')

        ax[0][0].legend(loc='lower left', fontsize=12)

        plt.subplots_adjust(wspace=0.05, hspace=0.3)

        if pngfile:
            plt.savefig(pngfile)    

    def qa_zwarn4(res):
        mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = gethist('RMAG', res)

        fig, ax = plt.subplots()
        ax.errorbar(mm, e1, ee1, fmt='o', label='ZWARN=0, $dz/(1+z)<0.003$')
        ax.errorbar(mm, e5, ee5, fmt='o', label='ZWARN=0 or ZWARN=4, $dz/(1+z)<0.003$')
        ax.axhline(y=1, ls='--', color='k')
        ax.grid()
        ax.set_xlabel(r'$r_{DECaLS}$ (AB mag)')
        ax.set_ylabel('Redshift Efficiency')
        ax.legend(loc='lower left')
        ax.set_ylim([0, 1.1])    

    def zstats(res):
        z = res['ZTRUE']
        dz = (res['Z'] - z)
        dzr = dz / (1 + z)
        s1 = (res['ZWARN'] == 0)
        s2 = (res['ZWARN'] == 0) & (np.abs(dzr) < 0.003)
        s3 = (res['ZWARN'] == 0) & (np.abs(dzr) >= 0.003)
        s4 = (res['ZWARN'] != 0)
        s5 = np.logical_and( np.logical_or( (res['ZWARN'] == 0), (res['ZWARN'] == 4) ), (np.abs(dzr) < 0.003) )

        return z, dz, dzr, s1, s2, s3, s4, s5

    def gethist(quantity, res, range=None):
        """Generate the histogram (and Poisson uncertainty) for various 
        sample cuts.  See zstats() for details.

        """
        var = res[quantity]
        z, dz, dzr, s1, s2, s3, s4, s5 = zstats(res)    

        h0, bins = np.histogram(var, bins=100, range=range)
        hv, _ = np.histogram(var, bins=bins, weights=var)
        h1, _ = np.histogram(var[s1], bins=bins)
        h2, _ = np.histogram(var[s2], bins=bins)
        h3, _ = np.histogram(var[s3], bins=bins)
        h4, _ = np.histogram(var[s4], bins=bins)
        h5, _ = np.histogram(var[s5], bins=bins)

        good = h0 > 2
        hv = hv[good]
        h0 = h0[good]
        h1 = h1[good]
        h2 = h2[good]
        h3 = h3[good]
        h4 = h4[good]
        h5 = h5[good]

        vv = hv / h0

        def _eff(k, n):
            eff = k / (n + (n==0))
            efferr = np.sqrt(eff * (1 - eff)) / np.sqrt(n + (n == 0))
            return eff, efferr

        e1, ee1 = _eff(h1, h0)
        e2, ee2 = _eff(h2, h0)
        e3, ee3 = _eff(h3, h0)
        e4, ee4 = _eff(h4, h0)
        e5, ee5 = _eff(h5, h0)

        return vv, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5

    def bgs_write_simdata(sim, rand, overwrite=False):
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
        simdata['AIRMASS'] = ref_obsconditions['AIRMASS']
        simdata['SEEING'] = ref_obsconditions['SEEING']
        simdata['MOONALT'] = ref_obsconditions['MOONALT']

        if 'moonfracmin' in sim.keys():
            simdata['MOONFRAC'] = rand.uniform(sim['moonfracmin'], sim['moonfracmax'], sim['nsim'])
        else:
            simdata['MOONFRAC'] = ref_obsconditions['MOONFRAC']

        if 'moonsepmin' in sim.keys():
            simdata['MOONSEP'] = rand.uniform(sim['moonsepmin'], sim['moonsepmax'], sim['nsim'])
        else:
            simdata['MOONSEP'] = ref_obsconditions['MOONSEP']

        if 'exptime' in sim.keys():
            simdata['EXPTIME'] = rand.uniform(sim['exptimemin'], sim['exptimemax'], sim['nsim'])
        else:
            simdata['EXPTIME'] = ref_obsconditions['EXPTIME']

        if overwrite or not os.path.isfile(simdatafile):     
            print('Writing {}'.format(simdatafile))
            write_bintable(simdatafile, simdata, extname='SIMDATA', clobber=True)

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

    def write_templates(outfile, flux, wave, meta):
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

    def bgs_sim_spectra(sim, overwrite=False, verbose=False):
        """Generate spectra for a given set of simulation parameters with 
        the option of overwriting files.

        """
        from desisim.scripts.quickspectra import sim_spectra

        rand = np.random.RandomState(sim['seed'])
        BGSmaker = BGStemplates(rand=rand, verbose=verbose)

        # Generate the observing conditions table.
        simdata = bgs_write_simdata(sim, rand, overwrite=overwrite)

        for ii, simdata1 in enumerate(simdata):

            # Generate the observing conditions dictionary.  
            obs = simdata2obsconditions(simdata1)

            # Generate the rest-frame templates.  Currently not writing out the rest-frame 
            # templates but we could.
            flux, wave, meta = bgs_make_templates(sim, rand, BGSmaker)

            truefile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}-true.fits'.format(sim['suffix'], ii))
            if overwrite or not os.path.isfile(truefile):    
                write_templates(truefile, flux, wave, meta)

            spectrafile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}.fits'.format(sim['suffix'], ii))
            if overwrite or not os.path.isfile(spectrafile):
                sim_spectra(wave, flux, 'bgs', spectrafile, obsconditions=obs, 
                            sourcetype='bgs', seed=sim['seed'], expid=ii)
            else:
                print('File {} exists...skipping.'.format(spectrafile))

    def bgs_redshifts(sim, overwrite=False):
        """Fit for the redshifts.

        """
        from redrock.external.desi import rrdesi    

        for ii in range(sim['nsim']):
            zbestfile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}-zbest.fits'.format(sim['suffix'], ii))
            spectrafile = os.path.join(simdir, sim['suffix'], 'bgs-{}-{:03}.fits'.format(sim['suffix'], ii))

            if overwrite or not os.path.isfile(zbestfile):
                rrdesi(options=['--zbest', zbestfile, '--ncpu', str(nproc), spectrafile])
            else:
                print('File {} exists...skipping.'.format(zbestfile))    

    def bgs_gather_results(sim, overwrite=False):
        """Gather all the pieces so we can make plots.

        """
        from desispec.io.spectra import read_spectra
        from desispec.io.zfind import read_zbest

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
                zbest = read_zbest(zbestfile)
                # Assume the tables are row-ordered!
                result['Z'][nspec*ii:nspec*(ii+1)] = zbest.z
                result['ZERR'][nspec*ii:nspec*(ii+1)] = zbest.zerr
                result['ZWARN'][nspec*ii:nspec*(ii+1)] = zbest.zwarn

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

        
        
        
        
 
