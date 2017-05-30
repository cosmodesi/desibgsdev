from __future__ import print_function
import numpy as np
from scipy.special import gamma, gammaincc
from scipy.interpolate import splev, splrep
from scipy.optimize import brentq

class target_LF:

    '''
    Class containing methods for calculating the target luminosity
    function of the MXXL mock catalogue
    '''
    
    def __init__(self, version='v0.0.4'):

        '''
        Initilise target_LF object. version is the version number of the 
        MXXL mock ('v0.0.3' or 'v0.0.4')
        '''

        # GAMA LF Schechter parameters (from Loveday 2012)
        self.M_star   = -20.70  # M* - 5logh
        self.Phi_star = 9.4e-3  # (Mpc/h)^-3
        self.alpha    = -1.23

        # LF evolution parameters
        self.P = 1.8
        self.Q = 0.7 
        
        # Tabulated values of SDSS target cumulative LF (at z=0.1)
        mag_sdss, logn_sdss = np.loadtxt("sdss_target_%s.dat"%version,
                                         unpack=True)

        # Spline fit to SDSS target LF
        tck = splrep(mag_sdss, logn_sdss)
        self.Phi_cum_sdss_z01 = lambda mag: splev(mag, tck)


    def Phi_cum_gama(self, mag, z):
        """
        Cumulative Schechter luminosity function from GAMA evolved
        to redshift z with evolution parameters P and Q
        """
        M_star_z = self.M_star - self.Q * (z-0.1)
        Phi_star_z = self.Phi_star * 10**(0.4*self.P*z)
        a = self.alpha
        t = 10**(0.4 * (M_star_z-mag))
        n = Phi_star_z*(gammaincc(a+2, t)*gamma(a+2) - \
                              t**(a+1)*np.exp(-t)) / (a+1)
        return n


    def Phi_cum_sdss(self, mag, z):
        """
        Cumulative luminosity function from SDSS evolved to redshift z
        with evolution parameters P and Q

        The bright end of our SDSS luminosity function comes from
        integrating the HODs * MXXL mass function (at z=0.1). At the 
        faint end, we transition to the LF of Blanton 2003, then 
        extrapolate to fainter magnitudes using a power law.
        """

        # shift mags to z=0.1
        mag01 = mag + self.Q * (z - 0.1)

        # get number density at z=0.1
        logn01 = self.Phi_cum_sdss_z01(mag01)

        # shift back to redshift z
        logn = logn01 + 0.4 * self.P * (z - 0.1)

        return 10**logn
    

    def w(self, z):
        """
        Sigmoid function which describes the smooth transition 
        between the SDSS target luminsity function at z < 0.1, 
        and the GAMA target luminosity function at z > 0.2

        w(z=0.1) ~ 0
        w(z=0.2) ~ 1
        """
        return 1. / (1. + np.exp(-100*(z-0.15)))


    def Phi_cum(self, mag, z):
        """
        Cumulative luminosity function of the MXXL HOD mock catalogue.
        This transitions from SDSS at z < 0.1 to GAMA at z > 0.2
        """
        W = self.w(z)
        return (1-W) * self.Phi_cum_sdss(mag, z) + W * self.Phi_cum_gama(mag, z)


    def f_root(self, mag, n, z):
        """
        Root finding function used in n_to_mag(n,z)
        """
        return self.Phi_cum(mag, z) - n


    def n_to_mag(self, n, z):
        """
        Returns the absolute magnitude threshold (^0.1M_r - 5logh)
        for which the sample of galaxies brighter than this at 
        redshift z has a number density n (in (Mpc/h)^-3)
        """
        return brentq(self.f_root, -16, -24, args=(n, z))


    
if __name__ == "__main__":
    
    # Test code
    
    # for number density 1e-3 (Mpc/h)^-3, print magnitude 
    # threshold at different redshifts
    
    n = 1e-3
    z = np.arange(0, 0.61, 0.1)
    mag = np.zeros(len(z))

    print("n=%.2e (Mpc/h)^-3" %n)

    lf = target_LF(version='v0.0.4')
    
    for i in range(len(z)):
        mag[i] = lf.n_to_mag(n, z[i])
        print("z=%.1f, Mr=%.2f" %(z[i], mag[i]))
