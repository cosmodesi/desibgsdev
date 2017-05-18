import numpy as np
import os.path, sys
import h5py
from scipy.interpolate import splev, splrep
from randoms import make_random_catalogue,make_Nrandom_catalogue

#############################
#
#  Input ARGUMENTS
#
narg = len(sys.argv)
if(narg == 7):
    mag_lim = float(sys.argv[1])
    N_rand = int(sys.argv[2])
    version = sys.argv[3]
    singlef = sys.argv[4]
    Nz_path = sys.argv[5]
    outdir = sys.argv[6]
else:
    sys.exit('6 arguments to be passed')

singlefile = False
if(singlef=='True' or singlef=='true'):
    singlefile = True

if (singlefile):
    # Generate ONE hdf5 random catalogue with N_rand*Ngals
    root = "randoms_r%.1f_N%i" %(mag_lim, N_rand)
    file_hdf5 = outdir+root+"_singlefile.hdf5"
    file_ascii = outdir+root+"_singlefile.txt"
    make_random_catalogue(mag_lim, N_rand, version, Nz_path, file_hdf5)

    # Transform into ASCII
    if (os.path.isfile(file_hdf5)):
        f = h5py.File(file_hdf5,'r')
        ra   = f["ra"].value
        dec  = f["dec"].value
        z    = f["z_obs"].value
        f.close()

        #Write the information into an ascii file
        tofile = zip(ra,dec,z)
        with open(file_ascii, 'w') as outf:                            
            outf.write('# ra,dec,z \n')                    
            np.savetxt(outf,tofile,fmt=('%.8f'))    
            outf.closed             
            print 'Output: ',file_ascii
    else:
        print 'NOT found:', file_hdf5

else:
    # Generate N_rand hdf5 random catalogues 
    file_name = "randoms_r%.1f_N%i_" %(mag_lim, N_rand)
    root = outdir+file_name
    make_Nrandom_catalogue(mag_lim, N_rand, version, Nz_path, root)

    for i in range(N_rand):
        file_name = root+str(i+1)+".hdf5"

        # Transform into ASCII
        file_ascii = root+str(i+1)+".txt"
        if (os.path.isfile(file_name)):
            f = h5py.File(file_name,'r')
            ra   = f["ra"].value
            dec  = f["dec"].value
            z    = f["z"].value
            f.close()

            #Write the information into an ascii file
            tofile = zip(ra,dec,z)
            with open(file_ascii, 'w') as outf:                            
                outf.write('# ra,dec,z \n')                    
                np.savetxt(outf,tofile,fmt=('%.8f'))    
                outf.closed             
                print 'Output: ',file_ascii
        else:
            print 'NOT found:', file_name

    
    
