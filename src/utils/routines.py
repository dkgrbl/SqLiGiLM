#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####


import numpy as np
import scipy as sp

LIGHTSPEED_CM_THZ = 2.99792458e-2   #  cm * THz  

'''  Functions ''' 

def phys_sinc(x):
    return np.sinc(x / np.pi)


def nu_to_um(nu):
    return LIGHTSPEED_CM_THZ / nu * 1e4


def F_pump_gaussian_func(DeltaNu, tau_fwhm):
    # Gaussian function:
    # DeltaNu: detuning on nu-grid from   
    # tau_fwhm: FWHM is taken on squared field i.e. for intensity  
    DeltaOmega = 2. * np.pi * (-DeltaNu)
    F = np.exp(-(tau_fwhm * DeltaOmega)**2. / (8. * 0.6931)) 
    return F



''' grid class:  provides uniform grid in frequency range ''' 

class c_grid():
    def __init__(self, num_points, wave_central, nu_ratio):
        self.size = num_points 
        self.nu_ratio = nu_ratio

        self.wave_central = wave_central                     #  cm
        self.nu_central = LIGHTSPEED_CM_THZ / wave_central   #  THz 
        
        dnu_max = self.nu_central * nu_ratio

        self.dnu = np.linspace(-dnu_max, dnu_max, num_points)
        self.nu = self.dnu + self.nu_central
        self.wave =  LIGHTSPEED_CM_THZ / self.nu 
        self.wave_nm =  LIGHTSPEED_CM_THZ / self.nu * 1e7
        self.wave_um =  LIGHTSPEED_CM_THZ / self.nu * 1e4
        self.dnu_step = self.dnu[1] - self.dnu[0]

    @classmethod
    def init_with_step(cls, num_points, wave_central, dnu_step):
        nu_central = LIGHTSPEED_CM_THZ / wave_central   #  THz 
        dnu_max = 0.5 * (num_points-1) *  dnu_step 
        nu_ratio = dnu_max / nu_central  
        grid = cls(num_points, wave_central, nu_ratio)
        return grid




