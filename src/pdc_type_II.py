#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####


import numpy as np
import scipy as sp

from src.utils.routines import *
from src.utils.correlations import *
from src.master_eq_solver import *

''' Main class ''' 

class PDC_type_II():
    """
        Class for prepare the grid for type-I PDC
    """
    def __init__(self, calc_parameters):

        self.NUMPOINTS_sig = calc_parameters["NUMPOINTS"]
        self.NUMPOINTS_idl = calc_parameters["NUMPOINTS"]

        self.length             = calc_parameters["length"] 
        ''' 
            Grid creation 
        '''
        wave_pump       = calc_parameters["wave_pump"]
        
        self.dnu_step = calc_parameters["dnu_step"]
        self.sig_grid = c_grid.init_with_step(self.NUMPOINTS_sig, wave_pump*2., self.dnu_step)
        self.idl_grid = c_grid.init_with_step(self.NUMPOINTS_idl, wave_pump*2., self.dnu_step)
        
        self.nu_pump_central    = LIGHTSPEED_CM_THZ/wave_pump
        self.nu_sig_central     = 0.5*self.nu_pump_central
        self.nu_idl_central     = 0.5*self.nu_pump_central
        ''' 
            Dispersions 
        '''
        self.n_p_func   = calc_parameters["n_p_func"]
        self.n_sig_func = calc_parameters["n_sig_func"]
        self.n_idl_func = calc_parameters["n_idl_func"]
        self.alpha_db_um_sig_func = calc_parameters["alpha_db_sig_func"]
        self.alpha_db_um_idl_func = calc_parameters["alpha_db_idl_func"]

        self.profile_pump_func = calc_parameters["profile_pump_func"]
        self.QPM_vector = 0

        if "QPM_vector" in calc_parameters.keys():
            self.QPM_vector = calc_parameters["QPM_vector"]

        self.k_sig_grid = np.empty(0)
        self.k_idl_grid = np.empty(0)
        
        self.v_group_pump_central = np.nan
        self.output_phases        = np.empty(0)  
        ''' 
            Two-photon amplitude 
        '''
        self.tpa            = np.empty(0)
        self.Delta_k        = np.empty(0)
        self.F_spectrum     = np.empty(0)
        ''' 
            Master eq solution: 
        '''
        self.C_adag_a  = np.empty(0)
        self.C_a_a     = np.empty(0)
        ''' 
            Compute Grids 
        '''
        self.calc_Delta_and_F_mesh()
        self.apply_filter()
        self.calc_TPA()
        


    def save_to_npz(self, FilePath):

        result_dict =  {
                            "type": "type_II",
                            "NUMPOINTS_sig": self.NUMPOINTS_sig,
                            "NUMPOINTS_idl": self.NUMPOINTS_idl,
                            "length": self.length,

                            "nu_pump_central": self.nu_pump_central,
                            "nu_sig_central": self.nu_sig_central,
                            "nu_idl_central": self.nu_idl_central,
                            
                            "dnu_sig_grid": self.sig_grid.dnu,
                            "dnu_idl_grid": self.idl_grid.dnu,
                            
                            "C_adag_a": self.C_adag_a,
                            "C_a_a": self.C_a_a,
                        }

        np.savez(FilePath, **result_dict)


    def calc_Delta_and_F_mesh(self):
        
        def kpump(nu):
            return 2. * np.pi * self.n_p_func(nu_to_um(nu)) * nu / LIGHTSPEED_CM_THZ

        def ksig(nu):
            return 2. * np.pi * self.n_sig_func(nu_to_um(nu)) * nu / LIGHTSPEED_CM_THZ
        
        def kidl(nu):
            return 2. * np.pi * self.n_idl_func(nu_to_um(nu)) * nu / LIGHTSPEED_CM_THZ

        def delta_k_func(dnu_s, dnu_i, nu_p):
            k_p = kpump(nu_p + dnu_s + dnu_i)
            k_s = ksig(0.5 * nu_p + dnu_s)
            k_i = kidl(0.5 * nu_p + dnu_i)
            DelK = k_p - k_s - k_i
            return DelK

        self.k_sig_grid     = ksig(self.nu_sig_central + self.sig_grid.dnu) 
        self.k_idl_grid     = kidl(self.nu_idl_central + self.idl_grid.dnu) 
        
        dnu_mesh_sig, dnu_mesh_idl  = np.meshgrid(self.sig_grid.dnu, self.idl_grid.dnu)
        self.F_spectrum     = self.profile_pump_func(dnu_mesh_sig+dnu_mesh_idl)
        self.Delta_k        = delta_k_func(dnu_mesh_sig, dnu_mesh_idl, self.nu_pump_central)

        if self.QPM_vector:
            if self.QPM_vector=="auto":
                self.QPM_vector = delta_k_func(0, 0, self.nu_pump_central)
                self.Delta_k = self.Delta_k - self.QPM_vector
            else:
                self.Delta_k = self.Delta_k - self.QPM_vector

        #####
        ##### Calculate self.output_phases
        #####                   

        del_nu          = self.nu_pump_central * 1e-5
        d_kpump         = kpump(self.nu_pump_central + del_nu) - kpump(self.nu_pump_central) 
        v_group_pump_central = 2. * np.pi * del_nu / d_kpump
        
        T = self.length / v_group_pump_central       ## central time for the new time-frame
        
        phi_sig = self.k_sig_grid * self.length  
        phi_idl = self.k_idl_grid * self.length 

        phi_shift_sig = 2*np.pi*self.sig_grid.dnu * T           ## change time-frame 
        phi_shift_idl = 2*np.pi*self.idl_grid.dnu * T           ## change time-frame 
        
        output_phases_sig  = np.exp(1.j* phi_sig) * np.exp(-1.j* phi_shift_sig)
        output_phases_idl  = np.exp(1.j* phi_idl) * np.exp(-1.j* phi_shift_idl)
        
        self.output_phases = np.zeros((self.NUMPOINTS_sig+self.NUMPOINTS_idl), dtype="complex")
        self.output_phases[0:self.NUMPOINTS_sig] = output_phases_sig
        self.output_phases[self.NUMPOINTS_sig:]  = output_phases_idl

    def apply_filter(self, ratio=0.94, power=15):

        self.filter_sig = np.exp(- (self.sig_grid.dnu)**(2*power)/(ratio*self.sig_grid.dnu[-1])**(2*power) )
        self.filter_idl = np.exp(- (self.idl_grid.dnu)**(2*power)/(ratio*self.idl_grid.dnu[-1])**(2*power) )
        self.filter_2d = np.outer(self.filter_idl, self.filter_sig)
        self.F_spectrum = self.F_spectrum*self.filter_2d

    def solve_master_eq(self, Gain):
        length = self.length
        N = self.NUMPOINTS_sig
        M = self.NUMPOINTS_idl
 
        ####
        #### Normalize Gamma
        ####
        _, SVals, _ = lg.svd(self.tpa)
        Gamma       = Gain / (SVals[0] * length)
        
        ####
        #### Prepare alpha_matrix  
        ####

        def alpha_dB_sig_func(nu):
            return self.alpha_db_um_sig_func(nu_to_um(nu))
        
        def alpha_dB_idl_func(nu):
            return self.alpha_db_um_idl_func(nu_to_um(nu))

        alphas_dB = np.zeros(N+M) 
        alphas_dB[0:N] = alpha_dB_sig_func(self.nu_sig_central + self.sig_grid.dnu)
        alphas_dB[N:]  = alpha_dB_idl_func(self.nu_idl_central + self.idl_grid.dnu)

        alphas_array = alphas_dB / (2*2.1714724095162565) 
        alphas_i, alphas_j = np.meshgrid(alphas_array, alphas_array)
        
        alphas_matrix = alphas_i + alphas_j

        ####
        #### Prepare Delta_k, F_spectrum
        ####
        F_matrix      = np.zeros((N+M,N+M), dtype="complex")
        Delta_matrix  = np.zeros((N+M,N+M), dtype="complex")
            
        Delta_matrix[0:N, N:] = self.Delta_k.T
        Delta_matrix[N:, 0:N] = self.Delta_k
        F_matrix[0:N, N:]     = self.F_spectrum.T
        F_matrix[N:, 0:N]     = self.F_spectrum

        ####
        #### Solve slow equations and change time frame.
        ####

        C_adag_a_slow, C_a_a_slow = solve_slow_master_eqs(length, Gamma, F_matrix, Delta_matrix, alphas_matrix, rtol=1e-7, atol=1e-12)
        self.C_adag_a, self.C_a_a = transfrom_correlators_U(C_adag_a_slow, C_a_a_slow, np.diag(self.output_phases))
        return self.C_adag_a, self.C_a_a

    def calc_TPA(self):
        B = self.F_spectrum
        C = phys_sinc(self.Delta_k * self.length / 2.)
        D = np.exp(1.0j * self.Delta_k * self.length / 2.)
        self.tpa = -1.0j * B * C * D
        return self.tpa 
 
