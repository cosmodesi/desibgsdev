
# generate_ascii_randoms


ascii
-----
This directory only contains read2ascii.py, a code to read the MXXL mock in hdf5 format and produce an ascii file for a given cut in apparent magnitude.

randoms
-------
Here you can find a modification of Alex Smith's code, taken from NERSC:
/project/projectdirs/desi/mocks/bgs/MXXL/randoms/randoms.py

Healpy and hdf5 libraries  needed in order to run the above script (see the run.
csh):

libs/hdf5/1.8.17 apps/python/2.7.8/gcc-4.4.7

* To run Alex's code, set in qsub.sh:
  singlefile='True'
  
  once you have modified the paths and names of the queues, aslo in qsub.sh, simply execute it:
  ./qsub.sh

* When you set in qsub.sh:
  singlefile='False'

  N_rand random files will be created, making use of the pygsl library to produc
e less correlated random numbers than in the case of simply using numpy. This code is not fast, as producing 20Ngal files took about 5h for Ngal=11493529 and 8h30min for Ngal=20869763
