import numpy as np
import h5py
import os.path,sys
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
from Cosmology import *

#-------------------Modify the followint lines if needed
hdf5 = True

mag_lim = 20.0 #19.5 
N_rand = 3 #20
version = 'v0.0.4'

# Mock name
nom = 'BGS_r20.6.hdf5'

# Cosmology from Angulo+12
set_cosmology(omega0=0.25,omegab=0.045,lambda0=0.75, \
                  h0=0.73, universe="Flat",include_radiation=False)

# Path to mocks and N(z)
pathin = '/mnt/lustre/desi/MXXL/nersc_download/'

# Path to randoms and ascii mock
rootrandom = '/mnt/lustre/desi/MXXL/catalogues/'
outdir = rootrandom+'randoms/'+version+'/'

# Bins for N(z) histogram
nbins = 81
#-------------------Modify until here----------------------

file_random = "randoms_r%.1f_N%i_" %(mag_lim, N_rand)
file_nz = pathin+'randoms/'+version+'/dNdz_r'+str(mag_lim)+'.dat'
if not(os.path.isfile(file_nz)):
    print 'STOP, N(z) not found:', file_nz ; sys.exit()
xnz, ynz = np.loadtxt(file_nz).transpose()

ra = [] ; dec = [] ; zz = []
if hdf5:
    file_mock = pathin+'desi_footprint/'+version+'/'+nom
    if not(os.path.isfile(file_mock)):
        print 'STOP, mock not found:', file_mock ; sys.exit()
    f = h5py.File(file_mock,'r')
    ra1_mock   = f["Data/ra"].value
    dec1_mock  = f["Data/dec"].value
    z1_mock    = f["Data/z_obs"].value
    app_mag = f["Data/app_mag"].value
    f.close()

    ind = np.where(app_mag<mag_lim)
    ra_mock = ra1_mock[ind]
    dec_mock = dec1_mock[ind]
    zz_mock = z1_mock[ind]

    for i in range(N_rand):
        random = outdir+file_random+str(i+1)+".hdf5"
        if not(os.path.isfile(random)):
            print 'STOP, not found:', random ; sys.exit()

        f = h5py.File(random,'r')
        ra  = np.append(ra,f["ra"].value)
        dec = np.append(dec,f["dec"].value)
        zz   = np.append(zz,f["z"].value)
        f.close()

else:
    file_mock = rootrandom+'/mocks/'+version+\
        '/BGS_r'+str(mag_lim)+'.txt'
    if not(os.path.isfile(file_mock)):
        print 'STOP, mock not found:', file_mock ; sys.exit()
    ra_mock,dec_mock,zz_mock = np.loadtxt(file_mock, \
                                             usecols=(0,1,2), unpack=True)

    for i in range(N_rand):
        random = outdir+file_random+str(i+1)+".txt"
        if not(os.path.isfile(random)):
            print 'STOP, not found:', random ; sys.exit()
        ra,dec,zz = np.loadtxt(random, unpack=True)

ngal = len(ra_mock)

# Check values of mocks and randoms
rlow = min(ra_mock) ; rhigh = max(ra_mock)
ind = np.where((ra>=rlow) & (ra<=rhigh))
rout = len(ra) - np.shape(ind)[1] 
print 'RAs outside mock(',rlow,',',rhigh,')=',rout    

dlow = min(dec_mock) ; dhigh = max(dec_mock)
ind = np.where((dec>=dlow) & (dec<=dhigh))
dout = len(dec) - np.shape(ind)[1]
print 'DECs outside mock(',dlow,',',dhigh,')=',dout

zlow = min(zz_mock) ; zhigh = max(zz_mock)
ind = np.where((zz>=zlow) & (zz<=zhigh))
zout = len(zz) - np.shape(ind)[1]
print 'zs outside mock(',zlow,',',zhigh,')=',zout

indout = np.where((ra<rlow) | (ra>rhigh) | \
                     (dec<dlow) | (dec>dhigh) | \
                     (zz<zlow) | (zz>zhigh))
nout = rout+dout+zout 
noutper = nout*100./len(ra)
perc = '{:.3%}'.format(noutper)
#print perc+' random outliers ', nout,np.shape(indout)[1]
#print ra[indout],dec[indout],zz[indout]

# Figure ####################################
fig = plt.figure(figsize=(20,20))

# Plot RA and DEC
ax =plt.subplot(321) 
ax.set_xlabel('RA') ; ax.set_ylabel('DEC')

low = min(rlow,min(ra))
high = max(rhigh,max(ra))
lh = high-low
xmin = low - 0.1*lh ; xmax = high + 0.1*lh
ax.set_xlim([xmin,xmax])

low = min(dlow,min(dec))
high = max(dhigh,max(dec))
lh = high-low
ymin = low - 0.1*lh ; ymax = high + 0.1*lh
ax.set_ylim([ymin,ymax])

# Downsample to plot
val = 10000
if (ngal > val):
    idx = np.arange(ngal) ;     #print idx
    np.random.shuffle(idx)
    ax.plot(ra_mock[idx[:val]],dec_mock[idx[:val]],'k.',label='Mock')

ntake = min(val,int(ngal/N_rand))
idx = np.arange(len(ra)) #;print ntake,np.shape(idx)
np.random.shuffle(idx)
ax.plot(ra[idx[:ntake]],dec[idx[:ntake]],\
            'r.',alpha=0.5,label='Random')

# Outliers
ax.plot(ra[indout],dec[indout],\
            'gs',alpha=0.5,\
            label=str(nout)+' random outliers ('+perc+')')

leg = plt.legend(loc=2)
leg.draw_frame(False)


# N(z)
ax =plt.subplot(323) 
ax.set_xlabel('z') ; ax.set_ylabel('Number, area=1')

#ax.hist(zz_mock, nbins, facecolor='k',label='Mock')
#ax.hist(z, nbins, facecolor='r',label='Random',alpha=0.5)
#ax.plot(xnz,ynz,'bo',label='File')

# N(z) with normalized area 
diff = [] 
for i in range(len(xnz)-1):
    val = xnz[i+1]-xnz[i] ; diff = np.append(diff,val)
dif0 = np.mean(diff)
if (len(set(diff)) == 1):
    dif0 = diff[0]
norm = sum(ynz)*dif0
ax.hist(zz_mock, nbins, facecolor='k',normed=True,label='Mock')
ax.hist(zz, nbins, facecolor='r',alpha=0.5,normed=True,label='Random')
ax.plot(xnz,ynz/norm,'bo',label='File')

leg = plt.legend(loc=1)
leg.draw_frame(False)

# N(z) with normalized maximum
ax =plt.subplot(325) 
ax.set_xlabel('z') ; ax.set_ylabel('N(z)/max(N)')

hist, bin_edges = np.histogram(zz_mock,bins=nbins)
bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
val = float(np.max(hist)) ; y = hist/val #; print val
ax.plot(bin_center,y,'k',label='Mock')

hist, bin_edges = np.histogram(zz,bins=nbins)
bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
val = float(np.max(hist)) ; y = hist/val #; print val
ax.plot(bin_center,y,'r',alpha=0.5,label='Random')

val = np.max(ynz) ; y = ynz/val #; print val
ax.plot(xnz,y,'bo',label='File')

leg = plt.legend(loc=1)
leg.draw_frame(False)

###################################
print'*** Cartesian coordinates'

# Check values of mocks and randoms in x,y,z
dz_mock = comoving_distance(zz_mock[:])
x_mock = dz_mock[:]*np.cos(ra_mock[:]*(np.pi/180.))*np.cos(dec_mock[:]*(np.pi/180.))  
y_mock = dz_mock[:]*np.cos(ra_mock[:]*(np.pi/180.))*np.sin(dec_mock[:]*(np.pi/180.))
z_mock = dz_mock[:]*np.sin(ra_mock[:]*(np.pi/180.))

dz = comoving_distance(zz[:])
x_ran = dz[:]*np.cos(ra[:]*(np.pi/180.))*np.cos(dec[:]*(np.pi/180.))  
y_ran = dz[:]*np.cos(ra[:]*(np.pi/180.))*np.sin(dec[:]*(np.pi/180.))
z_ran = dz[:]*np.sin(ra[:]*(np.pi/180.))

# Release memory
#ra_mock, dec_mock, ra, dec = [None for i in range(4)]

xlow = min(x_mock) ; xhigh = max(x_mock)
ind = np.where((x_ran>=xlow) & (x_ran<=xhigh))
xout = len(x_ran) - np.shape(ind)[1] 
print 'x outside mock(',xlow,',',xhigh,')=',xout    

ylow = min(y_mock) ; yhigh = max(y_mock)
ind = np.where((y_ran>=ylow) & (y_ran<=yhigh))
yout = len(y_ran) - np.shape(ind)[1]
print 'y outside mock(',ylow,',',yhigh,')=',yout

zlow = min(z_mock) ; zhigh = max(z_mock)
ind = np.where((z_ran>=zlow) & (z_ran<=zhigh))
zout = len(z_ran) - np.shape(ind)[1]
print 'z outside mock(',zlow,',',zhigh,')=',zout

rindout = np.where((x_ran<xlow) | (x_ran>xhigh) | \
                      (y_ran<ylow) | (y_ran>yhigh) | \
                      (z_ran<zlow) | (z_ran>zhigh))
nout = rout+dout+zout 
noutper = nout*100./len(x_ran)
perc = '{:.3%}'.format(noutper)
#print perc+' random outliers ',nout,np.shape(rindout)[1]
#print 'zz,dz,x,y,z for outliers'
#print zz[indout], dz[indout],x_ran[indout],y_ran[indout],z_ran[indout]
#print 'x,y,z outliers: ra,dec,z'
#print ra[rindout],dec[rindout],zz[rindout]

# xy
ax =plt.subplot(322) 
ax.set_xlabel('x (Mpc/h)') ; ax.set_ylabel('y (Mpc/h)')

# Downsample to plot
val = 10000
if (ngal > val):
    idx = np.arange(ngal) ;     #print idx
    np.random.shuffle(idx)
    ax.plot(x_mock[idx[:val]],y_mock[idx[:val]],'k.',label='Mock')

ntake = min(val,int(ngal/N_rand))
idx = np.arange(len(x_ran)) #;print ntake,np.shape(idx)
np.random.shuffle(idx)
ax.plot(x_ran[idx[:ntake]],y_ran[idx[:ntake]],\
            'r.',alpha=0.5,label='Random')
# Outliers
ax.plot(x_ran[indout],y_ran[indout],\
            'gs',alpha=0.5,label='ra,dec,z outliers')

ax.plot(x_ran[rindout],y_ran[rindout],\
            'cp',alpha=0.5,\
            label=str(nout)+' xyz outliers ('+perc+')')

leg = plt.legend(loc=2)
leg.draw_frame(False)

# xz
ax =plt.subplot(324) 
ax.set_xlabel('x (Mpc/h)') ; ax.set_ylabel('z (Mpc/h)')

# Downsample to plot
val = 10000
if (ngal > val):
    idx = np.arange(ngal) ;     #print idx
    np.random.shuffle(idx)
    ax.plot(x_mock[idx[:val]],z_mock[idx[:val]],'k.',label='Mock')

ntake = min(val,int(ngal/N_rand))
idx = np.arange(len(x_ran)) #;print ntake,np.shape(idx)
np.random.shuffle(idx)
ax.plot(x_ran[idx[:ntake]],z_ran[idx[:ntake]],\
            'r.',alpha=0.5,label='Random')

# Outliers
ax.plot(x_ran[indout],z_ran[indout],\
            'gs',alpha=0.5,label='ra,dec,z outliers')

ax.plot(x_ran[rindout],z_ran[rindout],\
            'cp',alpha=0.5,\
            label=str(nout)+' xyz outliers ('+perc+')')

leg = plt.legend(loc=2)
leg.draw_frame(False)

# yz
ax =plt.subplot(326) 
ax.set_xlabel('y (Mpc/h)') ; ax.set_ylabel('z (Mpc/h)')

# Downsample to plot
val = 1000
if (ngal > val):
    idx = np.arange(ngal) ;     #print idx
    np.random.shuffle(idx)
    ax.plot(y_mock[idx[:val]],z_mock[idx[:val]],'k.',label='Mock')

ntake = min(val,int(ngal/N_rand))
idx = np.arange(len(y_ran)) #;print ntake,np.shape(idx)
np.random.shuffle(idx)
ax.plot(y_ran[idx[:ntake]],z_ran[idx[:ntake]],\
            'r.',alpha=0.5,label='Random')

# Outliers
ax.plot(y_ran[indout],z_ran[indout],\
            'gs',alpha=0.5,label='ra,dec,z outliers')

ax.plot(y_ran[rindout],z_ran[rindout],\
            'cp',alpha=0.5,\
            label=str(nout)+' xyz outliers ('+perc+')')

leg = plt.legend(loc=2)
leg.draw_frame(False)

# Release memory
dz, x_mock, y_mock, z_mock, x_ran, y_ran, z_ran = [None for i in range(7)]

#########################
#plt.show()
# Save figure
if hdf5:
    plotfile = outdir+'check_'+file_random+'hdf5.png'
else:
    plotfile = outdir+'check_'+file_random+'txt.png'
fig.savefig(plotfile)
print 'Plot:', plotfile

