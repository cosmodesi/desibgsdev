#! /usr/bin/env python
import numpy as np
import h5py
import os.path, sys

# Modify the magnitude limit, version and file name if needed
maglim = 20.0

version = 'v0.0.4'
nom = 'BGS_r20.6.hdf5'
#############################################################

path1 = '/mnt/lustre/desi/MXXL/'
pathin = 'nersc_download/desi_footprint/'+version+'/'
pathou = 'catalogues/mocks/'+version+'/'

infile = path1+pathin+nom
if (os.path.isfile(infile)):
    f = h5py.File(infile,'r')
    ra1   = f["Data/ra"].value
    dec1  = f["Data/dec"].value
    zobs1 = f["Data/z_obs"].value
    zcos1 = f["Data/z_cos"].value
    app_mag = f["Data/app_mag"].value
    f.close()

    ind = np.where(app_mag<maglim)
    ra = ra1[ind]
    dec = dec1[ind]
    zobs = zobs1[ind]
    zcos = zcos1[ind]

    #Write the information into an ascii file
    output = path1+pathou+'BGS_r'+str(maglim)+'.txt'
    tofile = zip(ra,dec,zobs,zcos)
    with open(output, 'w') as outf:                            
        outf.write('# ra,dec,zobs,zcos \n')                    
        np.savetxt(outf,tofile,fmt=('%.8f'))    
        outf.closed             
    print 'Output: ',output
else:
    print 'NOT found:', infile
