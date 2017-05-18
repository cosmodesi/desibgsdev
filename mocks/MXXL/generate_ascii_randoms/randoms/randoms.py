import numpy as np
import os.path, sys
import h5py
from scipy.interpolate import splev, splrep

def get_z(version, mag_lim, N_rand, Nz_path):
    # generate random redshifts from dN/dz
    path = Nz_path+version
    infile = path+'/dNdz_r%.1f.dat'%mag_lim

    if (os.path.isfile(infile)):
        #read dN/dz from file
        z, dNdz = np.loadtxt(infile).transpose()
    
        z[1:] -= (z[1]/2.)    #get redshift bin centres
        tck = splrep(z, dNdz) #spline representation of dN/dz
    
        #cumulative dN/dz
        zs = np.arange(0,0.80000001, 0.000001)
        Ns = np.cumsum(splev(zs, tck))
        Ns = Ns / float(Ns[-1])

        # total number of galaxies in DESI footprint
        # for each magnitude limit
        if mag_lim == 20:
            Ngal = 20869763
        elif mag_lim == 19.5:
            Ngal = 11493529

        #draw random numbers
        N = np.random.rand(Ngal*N_rand)

        #search cumulative dN/dz
        ind = np.searchsorted(Ns, N)
        z = zs[ind]
    else:
        print 'STOP: Wrong N(z) file ',infile
        sys.exit()
    return z


def mask(ra,dec,Nz_path): #returns a logical array which is true if in DESI 
    import time as t
    import numpy as np
    import healpy as hp

    tile_diam=3.21 #  Diameter of DESI field of view
#
    t1=t.time()
#   Read all the tile centres and classifications
    tile_file=Nz_path+'desi-tiles-commented.ecsv'
    if (os.path.isfile(tile_file)):
        row_t = np.dtype([("tileid",np.int32),("ra",np.float32),("dec",np.float32),("pass",np.int32),("in",np.int32)])

        f = open(tile_file,'r')
        tiles=np.loadtxt(tile_file,dtype=row_t)
        f.close()

        tileid=tiles["tileid"]
        rat=tiles["ra"]
        dect=tiles["dec"]
        desi=tiles["in"]
        # Select the subset that define the DESI footprint
        dec_desi=dect[desi==1]
        ra_desi=rat[desi==1]
        tileid_desi=tileid[desi==1]
        # compute unit vectors for later ease of calculating dot products
        zhat_desi=np.sin(np.radians(dect[desi==1]))
        yhat_desi=np.cos(np.radians(dect[desi==1]))*np.sin(np.radians(rat[desi==1]))
        xhat_desi=np.cos(np.radians(dect[desi==1]))*np.cos(np.radians(rat[desi==1]))
        t2=t.time()
        print 'Time to read tiles: {0:.4f} '.format(t2-t1),' secs'


        t1=t.time()
        # Find the healpix index of each tile centre
        nside=16 #sets the fidelty of the healpix grid (pixels > tiles is required)
        npix=hp.nside2npix(nside)
        strad_pix=4.0*np.pi/float(npix)
        sqdeg_pix=strad_pix*(180/np.pi)**2
        diam_pix= np.sqrt(sqdeg_pix*4.0/np.pi)
        print "pixel size",sqdeg_pix,'sq deg pixel diameter=',diam_pix,' deg'
        theta=np.pi/2.0-np.radians(dec_desi)
        phi=2.0*np.pi-np.radians(ra_desi)
        ipix_desi=hp.ang2pix(nside,theta,phi) #tile pixel indices
        t2=t.time()
        print 'Time to index tiles: {0:.4f} '.format(t2-t1),' secs'

        # Sort tiles according to healpix index using a structured array
        t1=t.time()
        # Pack tile information into a structured array
        dt=np.dtype([('tileid',np.int,1),('ra',np.float,1),('dec',np.float,1),('xhat',np.float,1),('yhat',np.float,1),('zhat',np.float,1),('healpix',np.int,1),])
        tile=np.array(zip(tileid_desi,ra_desi,dec_desi,xhat_desi,yhat_desi,zhat_desi,ipix_desi),dtype=dt)
        tiles=np.sort(tile,order='healpix') #sort by healpix index
        print 'Time to sort tiles: {0:.4f} '.format(t2-t1),' secs'


        #  Compute unit vector components for all objects
        t1=t.time()
        cosdec=np.cos(np.radians(dec))
        xhat=cosdec*np.cos(np.radians(ra))
        yhat=cosdec*np.sin(np.radians(ra))
        zhat=np.sin(np.radians(dec))
        #   Find corresponding healpix
        theta=np.pi/2.0-np.arcsin(zhat)
        phi=2*np.pi-np.radians(ra)
        ipix=hp.ang2pix(nside,theta,phi)
        # Find neighbouring healpixels around each RA and dec
        nbrs=hp.get_all_neighbours(nside,theta,phi) # 8 neighbours
        nbrs=np.row_stack((nbrs,ipix.T)) #add in 9th central pixel
        t2=t.time()
        print 'Time to find neigbouring pixels: {0:.4f} '.format(t2-t1),' secs'


        t1=t.time()
        #Value of cos(theta) corresponding to a tile radius
        cosmin=np.cos(np.radians(tile_diam/2.0))
        #First apply quick geometric mask to set to false points we don't 
        #need to explicitly compare with the neighbouring tiles
        keep = sim_mask(ra,dec)
        # loop over each object and set keep=true if within any tile
        for i in range(0,ra.size,1):
            #if i%10000==0:
            #     print i, ra.size
            if (keep[i]): # skip objects already flagged false by sim_mask()
                # Find the indices bracketing the set of tiles in each of the 9 healpixels
                i1=np.searchsorted(tiles['healpix'],nbrs[:,i],side='left')
                i2=np.searchsorted(tiles['healpix'],nbrs[:,i],side='right')
                # Extract as a single array the tiles centred in these 9 healpixels
                tile_sel= np.concatenate((tiles[i1[0]:i2[0]],tiles[i1[1]:i2[1]],tiles[i1[2]:i2[2]],tiles[i1[3]:i2[3]],tiles[i1[4]:i2[4]],tiles[i1[5]:i2[5]],tiles[i1[6]:i2[6]],tiles[i1[7]:i2[7]],tiles[i1[8]:i2[8]]))
                xhat_sel=tile_sel['xhat'] #create views 
                yhat_sel=tile_sel['yhat'] #of the unit 
                zhat_sel=tile_sel['zhat'] #vectors of these tiles
                # keep if the dot product to any tile centre is greater than cosmin 
                keep[i]=np.any(xhat_sel*xhat[i]+yhat_sel*yhat[i]+zhat_sel*zhat[i]>cosmin)
    else:
        print 'STOP: Wrong tile file ',tile_file
        sys.exit()
    return keep 


def sim_mask(ra,dec):  
    #returns a logical array which is true if in approximate DESI footprint
    # pass in arrays of RA and dec both in degrees
    sindec=np.sin(dec*np.pi/180.0)
    # cuts enclosing the NGP region
    keep = np.logical_and(ra>117.0-32.0*sindec,sindec>-0.17)
    keep = np.logical_and(keep,ra<260.0+45*sindec) 
    # cuts enclosing the SGP region
    keep2=  np.logical_or(ra>303.0,ra<75.-30.0*sindec)
    keep2=  np.logical_and(keep2,sindec<0.5)
    keep2=  np.logical_and(keep2,sindec>-0.35)
    # flag as inside DESI if in either region
    keep = np.logical_or(keep,keep2)
    return keep



def make_random_catalogue(mag_lim, N_rand, version, Nz_path, file_name):

    # draw random redshifts from dN/dz
    z = get_z(version,mag_lim, N_rand, Nz_path)

    # draw random ra and dec
    N_tot = len(z)
    ra = np.random.rand(N_tot*2) * 360
    sin_dec = np.random.rand(N_tot*2) * 1.35 - 0.35
    dec = np.arcsin(sin_dec) * 180 / np.pi

    # keep ra and dec that are in DESI footprint
    ind = mask(ra, dec, Nz_path)
    ra = ra[ind][:N_tot]
    dec = dec[ind][:N_tot]

    # save random catalogue
    f = h5py.File(file_name, "a")
    f.create_dataset("z",   data=z,   compression="gzip", dtype="f4")
    f.create_dataset("ra",  data=ra,  compression="gzip", dtype="f4")
    f.create_dataset("dec", data=dec, compression="gzip", dtype="f4")
    f.close()


def make_Nrandom_catalogue(mag_lim, N_rand, version, Nz_path, root):
    import pygsl.rng
    #http://pygsl.sourceforge.net/reference/pygsl/module-pygsl.sf.html
    # generate random redshifts from dN/dz
    path = Nz_path+version
    infile = path+'/dNdz_r%.1f.dat'%mag_lim

    if (os.path.isfile(infile)):
        #read dN/dz from file
        z, dNdz = np.loadtxt(infile).transpose()
        z[1:] -= (z[1]/2.)    #get redshift bin centres
        tck = splrep(z, dNdz) #spline representation of dN/dz

        #cumulative dN/dz
        zs = np.arange(0,0.80000001, 0.000001)
        Ns = np.cumsum(splev(zs, tck))
        Ns = Ns / float(Ns[-1])

        # total number of galaxies in DESI footprint
        # for each magnitude limit
        if mag_lim == 20:
            Ngal = 20869763
        elif mag_lim == 19.5:
            Ngal = 11493529

        # initialize the random number generator
        my_ran0 = pygsl.rng.ran0() 

        # loop over the number of files to be produced
        for i in range(N_rand):
            file_name = root+str(i+1)+".hdf5" 
        
            #draw random numbers
            N = my_ran0.uniform(Ngal)
            #search cumulative dN/dz
            ind = np.searchsorted(Ns, N)
            z = zs[ind]

            # draw random ra and dec
            N_tot = len(z)

            ra = my_ran0.uniform(N_tot*2) * 360
            sin_dec = my_ran0.uniform(N_tot*2) * 1.35 - 0.35
            dec = np.arcsin(sin_dec) * 180 / np.pi

            # keep ra and dec that are in DESI footprint
            ind = mask(ra, dec, Nz_path)
            ra = ra[ind][:N_tot]
            dec = dec[ind][:N_tot]

            # save random catalogue
            f = h5py.File(file_name, "w")
            f.create_dataset("z",   data=z,   compression="gzip", dtype="f4")
            f.create_dataset("ra",  data=ra,  compression="gzip", dtype="f4")
            f.create_dataset("dec", data=dec, compression="gzip", dtype="f4")
            f.close()


if __name__ == '__main__':

    mag_lim = 20.0   # magnitude limit
    N_rand = 2       # number of randoms = N_rand * number galaxies in catalogue
    file_name = "randoms_N%i_r%.1f_100.hdf5" %(N_rand, mag_lim)

    make_random_catalogue(mag_lim, N_rand, file_name)
