#!/bin/bash
#PBS -l walltime=00:01:00
#PBS -l nodes=1

#-------------------Modify the followint lines if needed
mag_lim=(19.5 20.0)
N_rand=3 #20
version='v0.0.4'
singlefile='False'

# Path to the N(z) files
Nz_path=/mnt/lustre/desi/MXXL/nersc_download/randoms/

# Set path for output
path=/mnt/lustre/desi/MXXL/catalogues/randoms/
outdir=$path$version'/'
#-------------------Modify until here----------------------

for mag in ${mag_lim[@]} ; do
    # Set the path to the log files

    logpath=/mnt/lustre/$(whoami)/Junk/ran${mag}
    'rm' -f $logpath

    qsub -q sciama1.q -o $logpath -j oe run.sh -v mag=$mag,n_rand=$N_rand,version=$version,singlefile=$singlefile,Nz_path=$Nz_path,outdir=$outdir

    # Memory intensive
    #qsub -q himem.q -o $logpath -j oe run.sh -v mag=$mag,n_rand=$N_rand,outdir=$outdir

    # Testing
    #qsub -I run.sh -v mag=$mag,n_rand=$N_rand,version=$version,singlefile=$singlefile,Nz_path=$Nz_path,outdir=$outdir
    #./run.sh -v mag=$mag,n_rand=$N_rand,version=$version,singlefile=$singlefile,Nz_path=$Nz_path,outdir=$outdir
done

echo 'End of the script'