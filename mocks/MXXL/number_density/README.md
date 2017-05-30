
Alex Smith (a.m.j.smith@durham.ac.uk) May 2017

##### n_to_mag.py

n_to_mag.py contains a routine for converting a number density to an absolute magnitude threshold in the MXXL mock catalogue

This is an updated version of the code on NERSC in
/project/projectdirs/desi/mocks/bgs/MXXL/n_to_mag/n_to_mag.py
which has been modified to place the code in the class target_LF

This code also requires the files sdss_target_v0.0.3.dat and sdss_target_v0.0.4.dat, which are tabulated files of the luminosity function in versions 0.0.3 and 0.0.4 of the mock at z=0.1

Absolute magnitudes are rest-frame SDSS r-band magnitudes k-corrected to z=0.1 (with no evolutionary correction).

##### To use in Python

from n_to_mag import target_LF

lf = target_LF(version='v0.0.4') #version 'v0.0.3' or 'v0.0.4'

n = 1e-3 # number density in (Mpc/h)^-3

z = 0.2  # redshift

lf.n_to_mag(n, z) # convert number density to magnitude threshold

Note that n and z must be floats, and not arrays. Looping over large arrays will be slow. 
