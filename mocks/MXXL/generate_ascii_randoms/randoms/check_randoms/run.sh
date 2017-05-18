#!/bin/bash
cd ${PBS_O_WORKDIR}

exec='check_randoms.py'

# Ensure that we have the needed modules
source /etc/profile.d/modules.sh
module purge
module add apps/python/2.7.8/gcc-4.4.7 libs/hdf5/1.8.17

# Run
python $exec 

