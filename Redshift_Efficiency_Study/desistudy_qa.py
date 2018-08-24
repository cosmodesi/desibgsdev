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
from desiutil.log import get_logger, DEBUG
log = get_logger()

import seaborn as sns
sns.set(style='white', font_scale=1.1, palette='deep')

max_snr_r = 4
max_snr_b = 4

def get_all_sims_obs(sim_names):
    from desistudy import get_predefined_sim_dict,get_predefined_obs_dict
    all_sims, all_obsconds = [],[]
    for simname in sim_names:
        cur_sim = get_predefined_sim_dict(simname)
        all_sims.append(cur_sim)
        cur_obs = get_predefined_obs_dict(simname)
        all_obsconds.append(cur_obs)

    sims = np.atleast_1d(all_sims)
    conditions = np.atleast_1d(all_obsconds)
    return sims, conditions

def print_sim_params(current_sim_name, sim_names, sims, conditions):
    ## Find appropriate simulation and observing conditions
    cur_sim = sims[ current_sim_name == sim_names ][0]
    cur_obs = conditions[ current_sim_name == sim_names ][0]

    ## Print the key/value pairs that define the simulation and observing conditions
    print("Sim: {}".format(current_sim_name))
    print("\tSim Parameters: ")
    for key,val in cur_sim.items():
        print("\t\t{}: {}".format(key,val))
    print("\tObs Parameters: ")
    for key,val in cur_obs.items():
        print("\t\t{}: {}".format(key,val))
        
def print_file_locations(simdir,sims):
    print("In {}/ :".format(simdir))
    for sim in sims:
        print("\tIn {}/:".format(sim['suffix']))
        if os.path.exists(os.path.join(simdir,sim['suffix'])):
            for filename in os.listdir(os.path.join(simdir,sim['suffix'])):
                if filename.split('-')[-1] == 'results.fits':
                    print("\t\t   {}".format(filename))

def plot_subset(wave, flux, truth, nplot=16, ncol=4, these=None, \
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
            thisax.text(xtxt, ytxt, r'$T_{{eff}}$={:.0f} K'.format(truth['TEFF'][indx]), ha=ha,\
                 va='top', transform=thisax.transAxes, fontsize=13)
        else:
            thisax.text(xtxt, ytxt, 'z={:.3f}'.format(truth['TRUEZ'][indx]), ha=ha, \
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

        mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = geteffhist(xx, res)
        thisax.errorbar(mm, e1, ee1, fmt='o', label='ZWARN=0')
        thisax.errorbar(mm, e2, ee2, fmt='o', label='ZWARN=0, $dz/(1+z)<0.003$')
        thisax.errorbar(mm, e3, ee3, fmt='o', label='ZWARN=0, $dz/(1+z)\geq 0.003$')
        #thisax.errorbar(mm, e4, ee4, fmt='o', label='ZWARN>0')

        thisax.set_xlabel(label)
        if dolog:
            thisax.set_xscale('log')
            thisax.xaxis.set_major_formatter(ScalarFormatter())
        if 'SNR' in xx:
            thisax.set_xlim([0,max_snr_r])
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
    mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = geteffhist('RMAG', res)

    fig, ax = plt.subplots()
    ax.errorbar(mm, e2, ee2, fmt='o', label='ZWARN=0, $dz/(1+z)<0.003$')
    ax.errorbar(mm, e5, ee5, fmt='o', label='ZWARN=0 or ZWARN=4, $dz/(1+z)<0.003$')
    ax.axhline(y=1, ls='--', color='k')
    ax.grid()
    ax.set_xlabel(r'$r_{DECaLS}$ (AB mag)')
    ax.set_ylabel('Redshift Efficiency')
    ax.legend(loc='lower left')
    ax.set_ylim([0, 1.1])    

    
def qa_exptime(res,pngfile=None):
    """Redshift efficiency and S/N vs exptime

    """
    dolog = False
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
                                       
    ## Zeff vs Exptime
    thisax = ax[0]
    mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = geteffhist('EXPTIME', res)
    #thisax.errorbar(mm, e1, ee1, fmt='o', label='ZWARN=0')
    thisax.errorbar(mm, e2, ee2, fmt='o', label='ZWARN=0, $dz/(1+z)<0.003$')
    thisax.errorbar(mm, e3, ee3, fmt='o', label='ZWARN=0, $dz/(1+z)\geq 0.003$')
    thisax.errorbar(mm, e4, ee4, fmt='o', label='ZWARN>0')

    thisax.set_xlabel('Exposure Time')
    if dolog:
        thisax.set_xscale('log')
        thisax.xaxis.set_major_formatter(ScalarFormatter())

    thisax.axhline(y=1, ls='--', color='k')
    thisax.grid()
    thisax.set_ylim([0, 1.1])

    ## SNR vs Exptime
    thisax = ax[1]
    mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = gethist('SNR_R', res, cut_on='EXPTIME')
    #300-405,405-510,510-615,615-720
    thisax.errorbar(mm, e1, ee1, fmt='o', label='Exp=300-405s')
    thisax.errorbar(mm, e2, ee2, fmt='o', label='Exp=405-510s')
    thisax.errorbar(mm, e3, ee3, fmt='o', label='Exp=510-615s')
    thisax.errorbar(mm, e4, ee4, fmt='o', label='Exp=615-720s')

    thisax.set_xlabel(r'S/N [$r$ channel]')
    thisax.set_xlim([0,max_snr_r])
    if dolog:
        thisax.set_xscale('log')
        thisax.xaxis.set_major_formatter(ScalarFormatter())

    thisax.axhline(y=1, ls='--', color='k')
    thisax.grid()
    thisax.set_ylim([0, 1.1])     

    ax[0].set_ylabel('Redshift Efficiency')
    ax[0].legend(loc='best', fontsize=12)
    ax[1].legend(loc='best', fontsize=12)
    plt.subplots_adjust(wspace=0.05, hspace=0.3)

    if pngfile:
        plt.savefig(pngfile)
     
    ## Zeff vs SNR, exp cut
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    thisax = ax
    thisax.scatter(res['EXPTIME'], res['SNR_R'])
    #thisax.errorbar(mm, e1, ee1, fmt='o', label='ZWARN=0')
    
    thisax.set_xlabel('Exposure Time')
    thisax.set_ylabel(r'S/N [$r$ channel]')
    thisax.set_xlim(0,780)
    if dolog:
        thisax.set_xscale('log')
        thisax.xaxis.set_major_formatter(ScalarFormatter())

    thisax.grid()
        
def qa_moonfrac(res,pngfile=None):
    """Redshift efficiency and S/N vs moon fraction

    """
    dolog = False
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
                                       
    ## Zeff vs Exptime
    thisax = ax[0]
    mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = geteffhist('MOONFRAC', res)
    #thisax.errorbar(mm, e1, ee1, fmt='o', label='ZWARN=0')
    thisax.errorbar(mm, e2, ee2, fmt='o', label='ZWARN=0, $dz/(1+z)<0.003$')
    thisax.errorbar(mm, e3, ee3, fmt='o', label='ZWARN=0, $dz/(1+z)\geq 0.003$')
    thisax.errorbar(mm, e4, ee4, fmt='o', label='ZWARN>0')

    thisax.set_xlabel('Moon Fraction')
    if dolog:
        thisax.set_xscale('log')
        thisax.xaxis.set_major_formatter(ScalarFormatter())

    thisax.axhline(y=1, ls='--', color='k')
    thisax.grid()
    thisax.set_ylim([0, 1.1])

    ## SNR vs Exptime
    thisax = ax[1]
    mm, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5 = gethist('SNR_R', res, cut_on='MOONFRAC')
    #0.6-0.7,0.7-0.8,0.8-0.9,0.9-1.0
    thisax.errorbar(mm, e1, ee1, fmt='o', label='MoonFrac=0.6-0.7')
    thisax.errorbar(mm, e2, ee2, fmt='o', label='MoonFrac=0.7-0.8')
    thisax.errorbar(mm, e3, ee3, fmt='o', label='MoonFrac=0.8-0.9')
    thisax.errorbar(mm, e4, ee4, fmt='o', label='MoonFrac=0.9-1.0')

    thisax.set_xlabel(r'S/N [$r$ channel]')
    thisax.set_xlim([0,max_snr_r])
    if dolog:
        thisax.set_xscale('log')
        thisax.xaxis.set_major_formatter(ScalarFormatter())

    thisax.axhline(y=1, ls='--', color='k')
    thisax.grid()
    thisax.set_ylim([0, 1.1]) 
    

    ax[0].set_ylabel('Redshift Efficiency')
    ax[0].legend(loc='best', fontsize=12)
    ax[1].legend(loc='best', fontsize=12)
    plt.subplots_adjust(wspace=0.05, hspace=0.3)

    if pngfile:
        plt.savefig(pngfile)
     
    ## Zeff vs SNR, exp cut
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    thisax = ax
    thisax.scatter(res['MOONFRAC'], res['SNR_R'])
    #thisax.errorbar(mm, e1, ee1, fmt='o', label='ZWARN=0')
    
    thisax.set_xlabel('Moon Fraction (1=Full, 0=New)')
    thisax.set_ylabel(r'S/N [$r$ channel]')
    thisax.set_xlim(0.5,1)
    if dolog:
        thisax.set_xscale('log')
        thisax.xaxis.set_major_formatter(ScalarFormatter())

    thisax.grid()

def stats(res,cut_on):

    if cut_on == 'EXPTIME':
        #300-405,405-510,510-615,615-720
        s1 = (res['EXPTIME'] > 300) & (res['EXPTIME'] <= 405) 
        s2 = (res['EXPTIME'] > 405) & (res['EXPTIME'] <= 510) 
        s3 = (res['EXPTIME'] > 510) & (res['EXPTIME'] <= 615) 
        s4 = (res['EXPTIME'] > 615) & (res['EXPTIME'] <= 720) 
        s5 = (res['EXPTIME'] > 720) & (res['EXPTIME'] <= 825)
    elif cut_on == 'MOONFRAC':
        #0.6-0.7,0.7-0.8,0.8-0.9,0.9-1.0
        s1 = (res['MOONFRAC'] > 0.6) & (res['MOONFRAC'] <= 0.7) 
        s2 = (res['MOONFRAC'] > 0.7) & (res['MOONFRAC'] <= 0.8) 
        s3 = (res['MOONFRAC'] > 0.8) & (res['MOONFRAC'] <= 0.9) 
        s4 = (res['MOONFRAC'] > 0.9) & (res['MOONFRAC'] <= 1.0) 
        s5 = (res['MOONFRAC'] > 0.5) & (res['MOONFRAC'] <= 0.6)
    else:
        truth = np.zeros(len(res['EXPTIME'])).astype(bool)
        s1,s2,s3,s4,s5 = truth.copy(),truth.copy(),truth.copy(),truth.copy(),truth.copy()
    return s1, s2, s3, s4, s5

def geteffhist(xquantity, res, range=None):
    """Generate the histogram (and Poisson uncertainty) for various 
    sample cuts.  See zstats() for details.

    """
    var = res[xquantity]
    z, dz, dzr, s1, s2, s3, s4, s5 = zstats(res)    

    h0, bins = np.histogram(var, bins=100, range=range)
    hv, _ = np.histogram(var, bins=bins, weights=var)
    h1, _ = np.histogram(var[s1], bins=bins)
    h2, _ = np.histogram(var[s2], bins=bins)
    h3, _ = np.histogram(var[s3], bins=bins)
    h4, _ = np.histogram(var[s4], bins=bins)
    h5, _ = np.histogram(var[s5], bins=bins)

    # only use hist bins with data in them
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

def gethist(xquantity, res, range=None, vert_axis='Z',cut_on='ZWARN'):
    """Generate the histogram (and Poisson uncertainty) for various 
    sample cuts.  See zstats() for details.

    """
    var = res[xquantity]
    s1, s2, s3, s4, s5 = stats(res,cut_on=cut_on) 
    
    h0, bins = np.histogram(var, bins=100, range=range)
    hv, _ = np.histogram(var, bins=bins, weights=var)
    h1, _ = np.histogram(var[s1], bins=bins)
    h2, _ = np.histogram(var[s2], bins=bins)
    h3, _ = np.histogram(var[s3], bins=bins)
    h4, _ = np.histogram(var[s4], bins=bins)
    h5, _ = np.histogram(var[s5], bins=bins)
    
    z = res['ZTRUE']
    dz = (res['Z'] - z)
    dzr = dz / (1 + z)
    zsuccess = (np.abs(dzr) <= 0.003) & (res['ZWARN'] == 0)
    
    s1 = s1 & zsuccess
    s2 = s2 & zsuccess
    s3 = s3 & zsuccess
    s4 = s4 & zsuccess
    s5 = s5 & zsuccess
    
    hs1, _ = np.histogram(var[s1], bins=bins)
    hs2, _ = np.histogram(var[s2], bins=bins)
    hs3, _ = np.histogram(var[s3], bins=bins)
    hs4, _ = np.histogram(var[s4], bins=bins)
    hs5, _ = np.histogram(var[s5], bins=bins)    

    good = h0 > 2
    hv = hv[good]
    h0 = h0[good]
    h1 = h1[good]
    h2 = h2[good]
    h3 = h3[good]
    h4 = h4[good]
    h5 = h5[good]
    hs1 = hs1[good]
    hs2 = hs2[good]
    hs3 = hs3[good]
    hs4 = hs4[good]
    hs5 = hs5[good]
    
    vv = hv / h0

    def _eff(k, n):
        eff = k / (n + (n==0))
        efferr = np.sqrt(eff * (1 - eff)) / np.sqrt(n + (n == 0))
        return eff, efferr

    e1, ee1 = _eff(hs1, h1)
    e2, ee2 = _eff(hs2, h2)
    e3, ee3 = _eff(hs3, h3)
    e4, ee4 = _eff(hs4, h4)
    e5, ee5 = _eff(hs5, h5)

    return vv, e1, e2, e3, e4, e5, ee1, ee2, ee3, ee4, ee5