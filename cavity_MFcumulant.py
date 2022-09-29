
import math
from math import sqrt as sqrt

import numpy as np
from numpy.linalg import eig
from numpy import sin as sin
from numpy import cos as cos
from numpy import exp as exp

import os

#import scipy.sparse as sp
#from scipy.linalg import expm
#from scipy.sparse.linalg import expm as sp_expm
#from scipy.sparse.linalg import eigs
#from scipy.sparse import csc_matrix
#from scipy.sparse import lil_matrix
#from scipy.special import comb
from scipy.integrate import complex_ode

from sympy.physics.quantum.cg import CG
from sympy import S

from collections import defaultdict

import sys

import parameters as param



class Cavity_MFcumulant:
    
    """
    Multilevel atoms in a cavity.
    Dynamics using mean-field or cumulant equations of motion.
    Assumes symmetric manifold for now.
    Will add possibility of sampling initial conditions for TWA and BBGKY.
    
    """
    
    def __init__(self):
        
        
        ### ---------------
        ###     GEOMETRY
        ### ---------------
        # This is not very useful for now, except for the parameter self.Ntotal
        self.dim = 0
        self.Nlist = []
        
        if param.geometry=='alltoall':
            self.Nlist.append(param.Ntotal);
            self.dim = 1;
        else: print("\ncavity_MFCumulant/init: Wrong type of geometry chosen.\n")
    
        self.Ntotal=1
        for ii in range(self.dim): self.Ntotal *= self.Nlist[ii];
        
        
        
        ### ---------------
        ###  SPACE VECTORS
        ### ---------------
        self.r_i = [np.zeros(3) for _ in range(self.Ntotal) ]
        self.fill_ri()
        
        
        ### ---------------
        ###     CONSTANTS
        ### ---------------
        
        self.dummy_constants()
        
        
        ### ---------------
        ###     LEVELS
        ### ---------------
        self.levels_info = { 'Fe':param.Fe, 'deg_e':param.deg_e, 'start_e':param.start_e, 'Fg':param.Fg, 'deg_g':param.deg_g, 'start_g':param.start_g }
        self.eg_to_level = { 'g': [ bb for bb in range(param.deg_g) ] ,\
                             'e': [ param.deg_g+aa for aa in range(param.deg_e) ] }
        self.level_to_eg = {}       # inverse of eg_to_level
        for ll in self.eg_to_level:
            for aa in self.eg_to_level[ll]: self.level_to_eg[aa] = ll
        #self.Mgs = [ -param.Fg+mm for mm in range(param.deg_g) ]
        #self.Mes = [ -param.Fe+mm for mm in range(param.deg_e) ]
        self.Mgs = [ -param.Fg+param.start_g+mm for mm in range(param.deg_g) ]
        self.Mes = [ -param.Fe+param.start_e+mm for mm in range(param.deg_e) ]
        self.Ms = { 'g': self.Mgs, 'e': self.Mes }
        self.Qs = [0,-1,1]
        #[ expression for item in list if conditional ]
        
        
        
        ### ---------------
        ###    NUMBERING
        ### ---------------
        if param.method in ['MF','TWA']:
            self.index = {}
            self.gg_index = {}
            self.ee_index = {}
            self.ge_index = {}
            
            self.numbering_MF()
            self.n_auxvariables = 0
            
        if param.method in ['cumulant','bbgky']:
            self.index = {}     # master dictionary containing all indices
            self.numbering_cumulant()
            self.auxiliary_numbering_cumulant()
            #print(self.index)
            
        if False:
            self.create_lists_of_slices_of_indices()
            
        
        print("Number of phase space variables: %i"%(self.nvariables))
        
        
        
        ### -------------------------
        ###    CLEBSCH-GORDAN COEFF
        ### -------------------------
        self.cg = {}
        for mg in self.Mgs:
            for q in self.Qs:
                if mg+q in self.Mes and q not in param.clebsch_zero_q:
                    self.cg[(mg,q)] = float(CG(param.Fg, mg, 1, q, param.Fe, mg+q).doit())
        if param.Fg==0 and param.Fe==0: self.cg[(0,0)] = 1.0
        self.cg = defaultdict( lambda: 0, self.cg )     # Set all other Clebsch-Gordan coefficients to zero
        
        if False:
            self.create_lists_of_ClebschGordan()
        
        print(self.cg)
        
        
        
        ### -------------------------
        ###           TWA
        ### -------------------------
        if param.method in ['MF','cumulant']: self.iterations = 1
        else: self.iterations = param.iterations
        
        
        
        ### ---------------
        ###     MEMORY
        ### ---------------
        self.memory_variables = (self.nvariables+self.n_auxvariables) * self.iterations * 2 * 8 / 1024**3        # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
        print("\nMemory variables: %g Gb."%(self.memory_variables))
        

        
        
        
        
    def dummy_constants (self):
        
        self.numlevels = param.deg_e + param.deg_g      # Total number of internal electronic states per atom
        self.localhilbertsize = self.numlevels
        
    def fill_ri (self):
        """Fill out matrix of atom positions r_i"""
        #if (param.geometry=='alltoall'):
            # Do nothing
            
    def numbering_MF (self):
        """Assign array position number to each one-point function and create dictionary label->number"""
        ind = 0
        for aa in self.Mgs:
            for bb in self.Mgs:
                self.gg_index[(aa,bb)] = ind
                ind += 1
        for aa in self.Mes:
            for bb in self.Mes:
                self.ee_index[(aa,bb)] = ind
                ind += 1
        for aa in self.Mgs:
            for bb in self.Mes:
                self.ge_index[(aa,bb)] = ind
                ind += 1
        
        # Fill dummy variables
        dummy = ind
        ind +=1
        
        ### Make default dictionaries
        self.gg_index = defaultdict( lambda: dummy, self.gg_index )
        self.ee_index = defaultdict( lambda: dummy, self.ee_index )
        self.ge_index = defaultdict( lambda: dummy, self.ge_index )
        
        self.nvariables = ind

        
        ### Fill out master index dictionary
        self.index['gg'] = self.gg_index
        self.index['ee'] = self.ee_index
        self.index['ge'] = self.ge_index
        
        
        
        
        
    def numbering_cumulant (self):
        """
        Assign array position number to each one- and two-point function and create dictionary label->number
        NOTE: SHOULD I AUTOMATIZE THIS??? Alternative: dictionary of dictionaries, run over ['g','e'] (MAYBE EASIER TO CHANGE LATER, e.g. REMOVE VARIABLES)
        """
        # 1 pt
        self.gg_index = {}
        self.ee_index = {}
        self.ge_index = {}
        # 2 pt
        self.gggg_index = {}
        self.eeee_index = {}
        self.ggee_index = {}
        self.ggge_index = {}
        self.geee_index = {}
        self.gege_index = {}
        self.geeg_index = {}
        
        ind = 0
        # 1 pt functions
        for aa in self.Mgs:
            for bb in self.Mgs:
                self.gg_index[(aa,bb)] = ind
                ind += 1
        for aa in self.Mes:
            for bb in self.Mes:
                self.ee_index[(aa,bb)] = ind
                ind += 1
        for aa in self.Mgs:
            for bb in self.Mes:
                self.ge_index[(aa,bb)] = ind
                ind += 1
                
        # 2 pt functions
        for aa in self.Mgs:
            for bb in self.Mgs:
                for cc in self.Mgs:
                    for dd in self.Mgs:
                        self.gggg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mes:
            for bb in self.Mes:
                for cc in self.Mes:
                    for dd in self.Mes:
                        self.eeee_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mgs:
            for bb in self.Mgs:
                for cc in self.Mes:
                    for dd in self.Mes:
                        self.ggee_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mgs:
            for bb in self.Mgs:
                for cc in self.Mgs:
                    for dd in self.Mes:
                        self.ggge_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mgs:
            for bb in self.Mes:
                for cc in self.Mes:
                    for dd in self.Mes:
                        self.geee_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mgs:
            for bb in self.Mes:
                for cc in self.Mgs:
                    for dd in self.Mes:
                        self.gege_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mgs:
            for bb in self.Mes:
                for cc in self.Mes:
                    for dd in self.Mgs:
                        self.geeg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
                        
                        
        # Fill dummy variable
        dummy = ind
        ind +=1
        
        # Total number variables (for eom solver)
        self.nvariables = ind
        
        ### Make default dictionaries
        self.gg_index = defaultdict( lambda: dummy, self.gg_index )
        self.ee_index = defaultdict( lambda: dummy, self.ee_index )
        self.ge_index = defaultdict( lambda: dummy, self.ge_index )
        
        self.gggg_index = defaultdict( lambda: dummy, self.gggg_index )
        self.eeee_index = defaultdict( lambda: dummy, self.eeee_index )
        self.ggee_index = defaultdict( lambda: dummy, self.ggee_index )
        self.ggge_index = defaultdict( lambda: dummy, self.ggge_index )
        self.geee_index = defaultdict( lambda: dummy, self.geee_index )
        self.gege_index = defaultdict( lambda: dummy, self.gege_index )
        self.geeg_index = defaultdict( lambda: dummy, self.geeg_index )

        
        ### Fill out master index dictionary
        self.index['gg'] = self.gg_index
        self.index['ee'] = self.ee_index
        self.index['ge'] = self.ge_index
        
        self.index['gggg'] = self.gggg_index
        self.index['eeee'] = self.eeee_index
        self.index['ggee'] = self.ggee_index
        self.index['ggge'] = self.ggge_index
        self.index['geee'] = self.geee_index
        self.index['gege'] = self.gege_index
        self.index['geeg'] = self.geeg_index
        
        #print(self.index)
         
    
    def auxiliary_numbering_cumulant (self):
        """
        Create dictionary of indices (label->number) for auxiliary variables in cumulant equations, starting from last index of actual variables
        NOTE: SHOULD I AUTOMATIZE THIS??? Alternative: dictionary of dictionaries, run over ['g','e'] (MAYBE EASIER TO CHANGE LATER, e.g. REMOVE VARIABLES)
        """
        ind = self.nvariables
        
        ######
        ######      VARIABLES WITH ge IN DIFFERENT ORDERINGS
        ######
        
        # 1 pt extra
        self.eg_index = {}
        # 2 pt extra
        self.eegg_index = {}
        self.ggeg_index = {}
        self.gegg_index = {}
        self.eggg_index = {}
        self.egee_index = {}
        self.eege_index = {}
        self.eeeg_index = {}
        self.egeg_index = {}
        self.egge_index = {}
        
        # 1 pt extra
        for aa in self.Mes:
            for bb in self.Mgs:
                self.eg_index[(aa,bb)] = ind
                ind += 1
                
        # 2 pt extra
        for aa in self.Mes:
            for bb in self.Mes:
                for cc in self.Mgs:
                    for dd in self.Mgs:
                        self.eegg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mgs:
            for bb in self.Mgs:
                for cc in self.Mes:
                    for dd in self.Mgs:
                        self.ggeg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mgs:
            for bb in self.Mes:
                for cc in self.Mgs:
                    for dd in self.Mgs:
                        self.gegg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mes:
            for bb in self.Mgs:
                for cc in self.Mgs:
                    for dd in self.Mgs:
                        self.eggg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mes:
            for bb in self.Mgs:
                for cc in self.Mes:
                    for dd in self.Mes:
                        self.egee_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mes:
            for bb in self.Mes:
                for cc in self.Mgs:
                    for dd in self.Mes:
                        self.eege_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mes:
            for bb in self.Mes:
                for cc in self.Mes:
                    for dd in self.Mgs:
                        self.eeeg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mes:
            for bb in self.Mgs:
                for cc in self.Mes:
                    for dd in self.Mgs:
                        self.egeg_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        for aa in self.Mes:
            for bb in self.Mgs:
                for cc in self.Mgs:
                    for dd in self.Mes:
                        self.egge_index[(aa,bb,cc,dd)] = ind
                        ind += 1
        
        
        
        
        ######
        ######      VARIABLES WITH PI and SIGMA
        ######
        
        self.Pminus_index = ind       # Pi^-
        ind += 1
        self.Pplus_index = ind        # Pi^+
        ind += 1
        self.Sminus_index = ind       # Sigma^-
        ind += 1
        self.Splus_index = ind        # Sigma^+
        ind += 1
        
        
        self.gg_Pminus_index = {}
        self.ee_Pminus_index = {}
        self.ge_Pminus_index = {}
        self.eg_Pminus_index = {}
        
        self.gg_Sminus_index = {}
        self.ee_Sminus_index = {}
        self.ge_Sminus_index = {}
        self.eg_Sminus_index = {}
        
        self.Pplus_gg_index = {}
        self.Pplus_ee_index = {}
        self.Pplus_ge_index = {}
        self.Pplus_eg_index = {}
        
        self.Splus_gg_index = {}
        self.Splus_ee_index = {}
        self.Splus_ge_index = {}
        self.Splus_eg_index = {}
        
        # gg
        for aa in self.Mgs:
            for bb in self.Mgs:
                self.gg_Pminus_index[(aa,bb)] = ind
                ind += 1
                self.gg_Sminus_index[(aa,bb)] = ind
                ind += 1
                self.Pplus_gg_index[(aa,bb)] = ind
                ind += 1
                self.Splus_gg_index[(aa,bb)] = ind
                ind += 1
        # ee
        for aa in self.Mes:
            for bb in self.Mes:
                self.ee_Pminus_index[(aa,bb)] = ind
                ind += 1
                self.ee_Sminus_index[(aa,bb)] = ind
                ind += 1
                self.Pplus_ee_index[(aa,bb)] = ind
                ind += 1
                self.Splus_ee_index[(aa,bb)] = ind
                ind += 1
        # ge
        for aa in self.Mgs:
            for bb in self.Mes:
                self.ge_Pminus_index[(aa,bb)] = ind
                ind += 1
                self.ge_Sminus_index[(aa,bb)] = ind
                ind += 1
                self.Pplus_ge_index[(aa,bb)] = ind
                ind += 1
                self.Splus_ge_index[(aa,bb)] = ind
                ind += 1
        # eg
        for aa in self.Mes:
            for bb in self.Mgs:
                self.eg_Pminus_index[(aa,bb)] = ind
                ind += 1
                self.eg_Sminus_index[(aa,bb)] = ind
                ind += 1
                self.Pplus_eg_index[(aa,bb)] = ind
                ind += 1
                self.Splus_eg_index[(aa,bb)] = ind
                ind += 1
        
        
        # Fill dummy variable
        dummy = ind
        ind +=1
        
        # Total number of auxiliary variables
        self.n_auxvariables = ind - self.nvariables
        
        ### Make default dictionaries
        # ge correlators
        self.eg_index = defaultdict( lambda: dummy, self.eg_index )
        
        self.eegg_index = defaultdict( lambda: dummy, self.eegg_index )
        self.ggeg_index = defaultdict( lambda: dummy, self.ggeg_index )
        self.gegg_index = defaultdict( lambda: dummy, self.gegg_index )
        self.eggg_index = defaultdict( lambda: dummy, self.eggg_index )
        self.egee_index = defaultdict( lambda: dummy, self.egee_index )
        self.eege_index = defaultdict( lambda: dummy, self.eege_index )
        self.eeeg_index = defaultdict( lambda: dummy, self.eeeg_index )
        self.egeg_index = defaultdict( lambda: dummy, self.egeg_index )
        self.egge_index = defaultdict( lambda: dummy, self.egge_index )
        
        # Pi/Sigma correlators
        self.gg_Pminus_index = defaultdict( lambda: dummy, self.gg_Pminus_index )
        self.ee_Pminus_index = defaultdict( lambda: dummy, self.ee_Pminus_index )
        self.ge_Pminus_index = defaultdict( lambda: dummy, self.ge_Pminus_index )
        self.eg_Pminus_index = defaultdict( lambda: dummy, self.eg_Pminus_index )
        
        self.gg_Sminus_index = defaultdict( lambda: dummy, self.gg_Sminus_index )
        self.ee_Sminus_index = defaultdict( lambda: dummy, self.ee_Sminus_index )
        self.ge_Sminus_index = defaultdict( lambda: dummy, self.ge_Sminus_index )
        self.eg_Sminus_index = defaultdict( lambda: dummy, self.eg_Sminus_index )
        
        self.Pplus_gg_index = defaultdict( lambda: dummy, self.Pplus_gg_index )
        self.Pplus_ee_index = defaultdict( lambda: dummy, self.Pplus_ee_index )
        self.Pplus_ge_index = defaultdict( lambda: dummy, self.Pplus_ge_index )
        self.Pplus_eg_index = defaultdict( lambda: dummy, self.Pplus_eg_index )
        
        self.Splus_gg_index = defaultdict( lambda: dummy, self.Splus_gg_index )
        self.Splus_ee_index = defaultdict( lambda: dummy, self.Splus_ee_index )
        self.Splus_ge_index = defaultdict( lambda: dummy, self.Splus_ge_index )
        self.Splus_eg_index = defaultdict( lambda: dummy, self.Splus_eg_index )
        
        
        ### Fill out master index dictionary
        # ge correlators
        self.index['eg'] = self.eg_index
        
        self.index['eegg'] = self.eegg_index
        self.index['ggeg'] = self.ggeg_index
        self.index['gegg'] = self.gegg_index
        self.index['eggg'] = self.eggg_index
        self.index['egee'] = self.egee_index
        self.index['eege'] = self.eege_index
        self.index['eeeg'] = self.eeeg_index
        self.index['egeg'] = self.egeg_index
        self.index['egge'] = self.egge_index
        
        
        # Pi/Sigma correlators
        self.index['Pminus'] = self.Pminus_index
        self.index['Pplus'] = self.Pplus_index
        self.index['Sminus'] = self.Sminus_index
        self.index['Splus'] = self.Splus_index
        
        self.index['ggPminus'] = self.gg_Pminus_index
        self.index['eePminus'] = self.ee_Pminus_index
        self.index['gePminus'] = self.ge_Pminus_index
        self.index['egPminus'] = self.eg_Pminus_index
        
        self.index['ggSminus'] = self.gg_Sminus_index
        self.index['eeSminus'] = self.ee_Sminus_index
        self.index['geSminus'] = self.ge_Sminus_index
        self.index['egSminus'] = self.eg_Sminus_index
        
        self.index['Pplusgg'] = self.Pplus_gg_index
        self.index['Pplusee'] = self.Pplus_ee_index
        self.index['Pplusge'] = self.Pplus_ge_index
        self.index['Ppluseg'] = self.Pplus_eg_index
        
        self.index['Splusgg'] = self.Splus_gg_index
        self.index['Splusee'] = self.Splus_ee_index
        self.index['Splusge'] = self.Splus_ge_index
        self.index['Spluseg'] = self.Splus_eg_index
        
     
    def create_lists_of_slices_of_indices(self):
        """
        Create dictionaries containing lists of several indices.
        For example, a list of all index['ge'][(aa,bb)].
        """
        self.allinds = {}
        
        temp = [[0,0]] + [[a,0] for a in [-2,-1,1,2]] + [[0,b] for b in [-2,-1,1,2]]
        
        self.allinds['gg'] = { (da,db): [ self.index['gg'][(aa+da,bb+db)] for aa in self.Ms['g'] for bb in self.Ms['g'] ] for da,db in temp }
        self.allinds['ge'] = { (da,db): [ self.index['ge'][(aa+da,bb+db)] for aa in self.Ms['g'] for bb in self.Ms['e'] ] for da,db in temp }
        self.allinds['eg'] = { (da,db): [ self.index['eg'][(aa+da,bb+db)] for aa in self.Ms['e'] for bb in self.Ms['g'] ] for da,db in temp }
        self.allinds['ee'] = { (da,db): [ self.index['ee'][(aa+da,bb+db)] for aa in self.Ms['e'] for bb in self.Ms['e'] ] for da,db in temp }
        
        print(self.allinds)
        
        
    def create_lists_of_ClebschGordan(self):
        """
        Create dictionaries containing lists of several Clebsch-Gordan coefficients.
        For example, a list of all cg[(a,q)].
        """
        self.allCG = {}
        
        self.allCG['gg'] = { ('a',d,q): np.array([ self.cg[(aa+d,q)] for aa in self.Ms['g'] for bb in self.Ms['g'] ]).reshape()  for d in [-1,0,1] for q in self.Qs }
        
        #self.allCG['a'] = { (la,lb,d,q): [ self.cg[(aa+d,q)] for aa in self.Ms[la] for bb in self.Ms[lb] ] \
        #                                 for la in ['g','e'] for lb in ['g','e'] for d in [-1,0,1] for q in self.Qs }
        
        
        
     
       
    def kron_del(self,a1,a2):
        """Kroenecker delta"""
        if a1==a2:
            return 1
        else:
            return 0
            
            
        
    '''   
        
            
    def get_array_position (self,indices):
        """Returns array position in row-order of a given lattice site (i1,i2,i3,...,in)"""
        if len(indices)!=self.dim: print("\nERROR/get_array_position: indices given to get_array_position have wrong length\n")
        
        array_position = indices[0]
        for ii in range(1,self.dim):
            array_position = array_position*self.Nlist[ii] + indices[ii]
        return array_position
    
    def get_indices (self,n):
        """Returns lattice indices (i1,i2,...) for a given array position"""
        indices = []
        temp = 0
        rest = n
        block = self.Ntotal
    
        while temp<self.dim:
            block = int( block/self.Nlist[temp] )
            indices.append(rest//block)     # should be able to do this more efficiently
            rest -= indices[temp]*block
            temp += 1
            
        return indices
       
        
    def multinomial(self,lst):
        """Computes multinomial (n1+n2+n3+...)!/(n1!n2!n3!...) for lst=[n1,n2,n3...]"""
        res, i = 1, sum(lst)
        i0 = lst.index(max(lst))
        for a in lst[:i0] + lst[i0+1:]:
            for j in range(1,a+1):
                res *= i
                res //= j
                i -= 1
        return res

    '''
        

####################################################################

#######                INITIAL CONDITIONS                ###########

####################################################################


    def choose_initial_condition (self,cIC=param.cIC):
        """Initialize density matrix to chosen initial condition."""
        #self.Sv = np.zeros((self.nvariables,1),dtype=complex)        # Later the "1" will become "iterations" for TWA
        self.Sv = np.zeros((self.nvariables,self.iterations),dtype=complex)        # Each column is a classical trajectory in TWA
        
        # Save operators associated to phase-space variables
        self.save_phase_space_ops()
        
        # Compute single-atom rho for given IC
        self.fill_single_p_rho()
        
        # Compute mean and/or variances (MF doesn't need variances, but it doesn't take too long to compute anyway)
        self.compute_initial_mean_and_variances()
        
        # Initialize phase-space variables and save in self.ps0 array
        if param.method in ['MF','cumulant']:
            self.ps0 = self.mean
        if param.method in ['TWA','bbgky']:
            if param.sampling=='gaussian':  self.sample_initial_state_gauss()
            else: print('\nERROR: param.sampling chosen not valid\n')
        
        # Transfer values of ps0 to Sv
        self.fill_initial_arrays_MFcumulant()
        
        
        # Perform rotation EDIT!!!!!!!!!!!!!!!!!!!!!!
        if param.rotate==True:
            self.rotate_IC()
        
        print(np.concatenate( (np.arange(self.nvariables).reshape((self.nvariables,1)), self.Sv) , axis=1) )
        print()
        #print(np.concatenate( (np.arange(self.nvariables+self.n_auxvariables).reshape((self.nvariables+self.n_auxvariables,1)), self.fill_auxiliary_variables(self.Sv)) , axis=1))
            
        #print(self.Sv)
            
        # Old MF initialization
        #optionsIC = { 'initialstate': self.IC_initialstate,
        #                'mixedstate': self.IC_mixedstate,
        #                'puresupstate': self.IC_puresupstate}
        #optionsIC[cIC]()
        
    
    def save_phase_space_ops (self):
        """ Save operators corresponding to phase-space variables and create dictionary numbering them for easy access """
        self.ps_Xindex = {}
        self.ps_Yindex = {}
        self.ps_Zindex = {}
        self.ps_ops = []
        
        ind = 0
        # sigma^x
        for aa in range(1,self.numlevels):
            for bb in range(aa):
                self.ps_ops.append( 0.5*(self.transitionOp_local(aa,bb)+self.transitionOp_local(bb,aa)) )
                self.ps_Xindex[(aa,bb)] = ind
                ind += 1
                
        # sigma^y
        for aa in range(1,self.numlevels):
            for bb in range(aa):
                self.ps_ops.append( -1j*0.5*(self.transitionOp_local(aa,bb)-self.transitionOp_local(bb,aa)) )
                self.ps_Yindex[(aa,bb)] = ind
                ind += 1
                
        # sigma^z
        for aa in range(self.numlevels):
            self.ps_ops.append( self.transitionOp_local(aa,aa) )
            self.ps_Zindex[aa] = ind
            ind += 1
            
        self.number_ps_ops = ind
        
        print(self.ps_Xindex)
        print(self.ps_Yindex)
        print(self.ps_Zindex)
        #print(self.ps_ops)
    
                
    
    def compute_initial_mean_and_variances (self):
        """ Computes mean and variances of phase-space observables defined in save_phase_space_ops. """
        
        # Compute single-atom mean and variances wrt phase-space operators
        self.mean = np.empty((self.number_ps_ops))
        self.variance = np.empty((self.number_ps_ops,self.number_ps_ops))
        
        for ii in range(self.number_ps_ops):
            self.mean[ii] = ( np.trace(self.ps_ops[ii] @ self.rho) ).real      #( (self.ps_ops[ii] @ self.rho).diagonal().sum() ).real
        for ii in range(self.number_ps_ops):        # Avoid computing this if no sampling
            for jj in range(ii+1):
                self.variance[ii,jj] = 0.5 * ( np.trace( (self.ps_ops[ii]@self.ps_ops[jj]+self.ps_ops[jj]@self.ps_ops[ii]) @ self.rho) ).real
        for ii in range(self.number_ps_ops):
            for jj in range(ii+1,self.number_ps_ops):
                self.variance[ii,jj] = self.variance[jj,ii]
                
        # Fill many-body covariance
        self.covariance = np.empty((self.number_ps_ops,self.number_ps_ops))
        for ii in range(self.number_ps_ops):        # Avoid computing this if no sampling
            for jj in range(self.number_ps_ops):
                self.covariance[ii,jj] = self.variance[ii,jj] - self.mean[ii]*self.mean[jj]
        self.covariance = self.covariance/self.Ntotal
                
        #print(self.mean)
        #print(self.variance)
        #print(self.covariance)
                
    
    def sample_initial_state_gauss (self):
        """
        Sample initial phase-space variables with multivariate gaussian random numbers.
        Mean and variance as computed in compute_initial_mean_and_variances.
        """
        
        self.ps0 = np.transpose( np.random.multivariate_normal(self.mean,self.covariance,self.iterations) )
        
        #self.ps0 = np.array( [ np.random.normal(self.mean[ii],sqrt(self.covariance[ii,ii]),self.iterations)  for ii in range(len(self.mean)) ] )
        
        
    
    
    def fill_initial_arrays_MFcumulant (self):
        """ Fill self.Sv array using data from self.ps0 """
        
        for ma in self.Mgs:
            for mb in self.Mgs:
                aa = self.eg_to_level['g'][self.Mgs.index(ma)]
                bb = self.eg_to_level['g'][self.Mgs.index(mb)]
                if aa>bb:   self.Sv[ self.gg_index[(ma,mb)] ] = self.ps0[ self.ps_Xindex[(aa,bb)] ] + 1j*self.ps0[ self.ps_Yindex[(aa,bb)] ]
                if aa<bb:   self.Sv[ self.gg_index[(ma,mb)] ] = self.ps0[ self.ps_Xindex[(bb,aa)] ] - 1j*self.ps0[ self.ps_Yindex[(bb,aa)] ]
                if aa==bb:  self.Sv[ self.gg_index[(ma,mb)] ] = self.ps0[ self.ps_Zindex[aa] ]
                
        for ma in self.Mes:
            for mb in self.Mes:
                aa = self.eg_to_level['e'][self.Mes.index(ma)]
                bb = self.eg_to_level['e'][self.Mes.index(mb)]
                if aa>bb:   self.Sv[ self.ee_index[(ma,mb)] ] = self.ps0[ self.ps_Xindex[(aa,bb)] ] + 1j*self.ps0[ self.ps_Yindex[(aa,bb)] ]
                if aa<bb:   self.Sv[ self.ee_index[(ma,mb)] ] = self.ps0[ self.ps_Xindex[(bb,aa)] ] - 1j*self.ps0[ self.ps_Yindex[(bb,aa)] ]
                if aa==bb:  self.Sv[ self.ee_index[(ma,mb)] ] = self.ps0[ self.ps_Zindex[aa] ]
                
        for ma in self.Mgs:
            for mb in self.Mes:
                aa = self.eg_to_level['g'][self.Mgs.index(ma)]
                bb = self.eg_to_level['e'][self.Mes.index(mb)]
                self.Sv[ self.ge_index[(ma,mb)] ] = self.ps0[ self.ps_Xindex[(bb,aa)] ] - 1j*self.ps0[ self.ps_Yindex[(bb,aa)] ]
        
        ### If cumulant, fill all other arrays
        if param.method in ['cumulant','']:
            
            oneminus1overN = 1-1/self.Ntotal
            oneoverN = 1/self.Ntotal
             
            # 2pt functions
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mgs:
                        for nn in self.Mgs:
                            self.Sv[ self.gggg_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.gg_index[(aa,bb)] ] * self.Sv[ self.gg_index[(mm,nn)] ] \
                                                                        + oneoverN * self.kron_del(bb,mm) * self.Sv[ self.gg_index[(aa,nn)] ]
            for aa in self.Mes:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mes:
                            self.Sv[ self.eeee_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ee_index[(aa,bb)] ] * self.Sv[ self.ee_index[(mm,nn)] ] \
                                                                        + oneoverN * self.kron_del(bb,mm) * self.Sv[ self.ee_index[(aa,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mes:
                        for nn in self.Mes:
                            self.Sv[ self.ggee_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.gg_index[(aa,bb)] ] * self.Sv[ self.ee_index[(mm,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mgs:
                        for nn in self.Mes:
                            self.Sv[ self.ggge_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.gg_index[(aa,bb)] ] * self.Sv[ self.ge_index[(mm,nn)] ] \
                                                                        + oneoverN * self.kron_del(bb,mm) * self.Sv[ self.ge_index[(aa,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mes:
                            self.Sv[ self.geee_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ge_index[(aa,bb)] ] * self.Sv[ self.ee_index[(mm,nn)] ] \
                                                                        + oneoverN * self.kron_del(bb,mm) * self.Sv[ self.ge_index[(aa,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mgs:
                        for nn in self.Mes:
                            self.Sv[ self.gege_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ge_index[(aa,bb)] ] * self.Sv[ self.ge_index[(mm,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mgs:
                            self.Sv[ self.geeg_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ge_index[(aa,bb)] ] * np.conj(self.Sv[ self.ge_index[(nn,mm)] ]) \
                                                                        + oneoverN * self.kron_del(bb,mm) * self.Sv[ self.gg_index[(aa,nn)] ]
                                                                        
                                                                        
        #if param.method in ['bbgky']:
        if param.method in ['bbgky']:
            
            #oneminus1overN = 1-1/self.Ntotal
            #oneover2N = 1/(2*self.Ntotal)
            
            oneminus1overN = 1
            oneover2N = 1/(2*self.Ntotal)
            sign = -1
             
            # 2pt functions
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mgs:
                        for nn in self.Mgs:
                            self.Sv[ self.gggg_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.gg_index[(aa,bb)] ] * self.Sv[ self.gg_index[(mm,nn)] ] \
                                                                        + oneover2N * self.kron_del(bb,mm) * self.Sv[ self.gg_index[(aa,nn)] ] \
                                                                        + sign * oneover2N * self.kron_del(aa,nn) * self.Sv[ self.gg_index[(mm,bb)] ]
            for aa in self.Mes:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mes:
                            self.Sv[ self.eeee_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ee_index[(aa,bb)] ] * self.Sv[ self.ee_index[(mm,nn)] ] \
                                                                        + oneover2N * self.kron_del(bb,mm) * self.Sv[ self.ee_index[(aa,nn)] ] \
                                                                        + sign * oneover2N * self.kron_del(aa,nn) * self.Sv[ self.ee_index[(mm,bb)] ]
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mes:
                        for nn in self.Mes:
                            self.Sv[ self.ggee_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.gg_index[(aa,bb)] ] * self.Sv[ self.ee_index[(mm,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mgs:
                        for nn in self.Mes:
                            self.Sv[ self.ggge_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.gg_index[(aa,bb)] ] * self.Sv[ self.ge_index[(mm,nn)] ] \
                                                                        + oneover2N * self.kron_del(bb,mm) * self.Sv[ self.ge_index[(aa,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mes:
                            self.Sv[ self.geee_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ge_index[(aa,bb)] ] * self.Sv[ self.ee_index[(mm,nn)] ] \
                                                                        + oneover2N * self.kron_del(bb,mm) * self.Sv[ self.ge_index[(aa,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mgs:
                        for nn in self.Mes:
                            self.Sv[ self.gege_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ge_index[(aa,bb)] ] * self.Sv[ self.ge_index[(mm,nn)] ]
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mgs:
                            self.Sv[ self.geeg_index[(aa,bb,mm,nn)] ] = oneminus1overN * self.Sv[ self.ge_index[(aa,bb)] ] * np.conj(self.Sv[ self.ge_index[(nn,mm)] ]) \
                                                                        + oneover2N * self.kron_del(bb,mm) * self.Sv[ self.gg_index[(aa,nn)] ] \
                                                                        + sign * oneover2N * self.kron_del(aa,nn) * self.Sv[ self.ee_index[(mm,bb)] ]
            
            
            
            
    
        
    def rotate_IC(self):
        """Rotate initial state with driving part for time t=1."""
        
        # Define mean-field equation
        def MF_eqs_IC (t,Svtemp):
            """Computes and returns right-hand side of mean-field equations"""
            Sv = Svtemp.reshape( (self.nvariables,self.iterations) )
            dSv = np.zeros((self.nvariables,self.iterations),dtype=complex)
        
            ############
            ####            IDEA: Avoid all the if clauses by defining all entries of Sv and CG that are out of bounds as 0
            ############
        
        
            # gg
            for aa in self.Mgs:
                for bb in self.Mgs:
                
                    # Driving
                    temp = 0
                    if aa in self.Mes: temp += param.Omet_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                    if bb in self.Mes: temp += - np.conj(param.Omet_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]
                    if aa-1 in self.Mes: temp += param.Omet_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])
                    if aa+1 in self.Mes: temp += param.Omet_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])
                    if bb-1 in self.Mes: temp += - np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]
                    if bb+1 in self.Mes: temp += - np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]
                    dSv[self.gg_index[(aa,bb)]] += 1j * temp
                
            # ee
            for aa in self.Mes:
                for bb in self.Mes:
                
                    # Driving
                    temp = 0
                    if bb in self.Mgs: temp += - param.Omet_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                    if aa in self.Mgs: temp += np.conj(param.Omet_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]
                    if bb+1 in self.Mgs: temp += - param.Omet_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])
                    if bb-1 in self.Mgs: temp += - param.Omet_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])
                    if aa+1 in self.Mgs: temp += np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]
                    if aa-1 in self.Mgs: temp += np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]
                    dSv[self.ee_index[(aa,bb)]] += 1j * temp
                
        
            # ge
            for aa in self.Mgs:
                for bb in self.Mes:
                
                    # Driving
                    temp = 0
                    if aa in self.Mes: temp += param.Omet_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]
                    if bb in self.Mgs: temp += - param.Omet_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]
                    if aa-1 in self.Mes: temp += param.Omet_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]
                    if aa+1 in self.Mes: temp += param.Omet_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]
                    if bb+1 in self.Mgs: temp += - param.Omet_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]
                    if bb-1 in self.Mgs: temp += - param.Omet_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]
                    dSv[self.ge_index[(aa,bb)]] += 1j * temp
        
        
            return dSv.reshape( (len(dSv)*len(dSv[0])) )
        
        
        
        # Define cumulant equation
        def cumulant_eqs_IC (t,Svtemp):
            """Computes and returns right-hand side of cumulant equations"""
            Sv = Svtemp.reshape( (self.nvariables,self.iterations) )
            dSv = np.zeros((self.nvariables,self.iterations),dtype=complex)
        
            # Compute arrays of auxiliary variables and add to Sv array
            # Note: After this step, the array Sv has size (self.nvariables+self.n_auxvariables , self.iterations),
            #       rows 0 to nvariables-1 correspond to main variables, the rest are auxiliary variables
            Sv = self.fill_auxiliary_variables(Sv)
        
            ###
            ###     1pt eqs
            ###
        
            # gg
            for aa in self.Mgs:
                for bb in self.Mgs:
                
                    # Driving
                    dSv[self.gg_index[(aa,bb)]] += 1j * ( param.Omet_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                          - np.conj(param.Omet_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                          + param.Omet_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])\
                                                          + param.Omet_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])\
                                                          - np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]\
                                                          - np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]\
                                                        )
                
            # ee
            for aa in self.Mes:
                for bb in self.Mes:
                
                    # Driving
                    dSv[self.ee_index[(aa,bb)]] += 1j * (- param.Omet_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                         + np.conj(param.Omet_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                         - param.Omet_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])\
                                                         - param.Omet_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])\
                                                         + np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]\
                                                         + np.conj(param.Omet_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]\
                                                        )
        
            # ge
            for aa in self.Mgs:
                for bb in self.Mes:
                
                    # Driving
                    dSv[self.ge_index[(aa,bb)]] += 1j * (param.Omet_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]\
                                                         - param.Omet_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]\
                                                         + param.Omet_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]\
                                                         + param.Omet_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]\
                                                         - param.Omet_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]\
                                                         - param.Omet_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]\
                                                        )
                
                
        
            ###
            ###     2pt eqs
            ###
        
            # gggg
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mgs:
                        for nn in self.Mgs:
                
                            # Driving
                            dSv[self.gggg_index[(aa,bb,mm,nn)]] += 1j * (param.Omet_Pi * ( self.cg[(aa,0)] * Sv[self.eggg_index[(aa,bb,mm,nn)]] \
                                                                                         + self.cg[(mm,0)] * Sv[self.ggeg_index[(aa,bb,mm,nn)]] )\
                                                                         - np.conj(param.Omet_Pi) * ( self.cg[(bb,0)] * Sv[self.gegg_index[(aa,bb,mm,nn)]] \
                                                                                                    + self.cg[(nn,0)] * Sv[self.ggge_index[(aa,bb,mm,nn)]] ) \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eggg_index[(aa-1,bb,mm,nn)]] \
                                                                                                      + self.cg[(aa,1)] * Sv[self.eggg_index[(aa+1,bb,mm,nn)]] ) \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(mm,-1)] * Sv[self.ggeg_index[(aa,bb,mm-1,nn)]] \
                                                                                                      + self.cg[(mm,1)] * Sv[self.ggeg_index[(aa,bb,mm+1,nn)]] ) \
                                                                         - np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(bb,-1)] * Sv[self.gegg_index[(aa,bb-1,mm,nn)]] \
                                                                                                               + self.cg[(bb,1)] * Sv[self.gegg_index[(aa,bb+1,mm,nn)]] ) \
                                                                         - np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(nn,-1)] * Sv[self.ggge_index[(aa,bb,mm,nn-1)]] \
                                                                                                               + self.cg[(nn,1)] * Sv[self.ggge_index[(aa,bb,mm,nn+1)]] ) \
                                                                )
                        
            # eeee
            for aa in self.Mes:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mes:
                
                            # Driving
                            dSv[self.eeee_index[(aa,bb,mm,nn)]] += 1j * ( - param.Omet_Pi * ( self.cg[(bb,0)] * Sv[self.egee_index[(aa,bb,mm,nn)]] \
                                                                                            + self.cg[(nn,0)] * Sv[self.eeeg_index[(aa,bb,mm,nn)]] )\
                                                                         + np.conj(param.Omet_Pi) * ( self.cg[(aa,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] \
                                                                                                    + self.cg[(mm,0)] * Sv[self.eege_index[(aa,bb,mm,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.egee_index[(aa,bb+1,mm,nn)]] \
                                                                                                      + self.cg[(bb-1,1)] * Sv[self.egee_index[(aa,bb-1,mm,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.eeeg_index[(aa,bb,mm,nn+1)]] \
                                                                                                      + self.cg[(nn-1,1)] * Sv[self.eeeg_index[(aa,bb,mm,nn-1)]] ) \
                                                                         + np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(aa+1,-1)] * Sv[self.geee_index[(aa+1,bb,mm,nn)]] \
                                                                                                               + self.cg[(aa-1,1)] * Sv[self.geee_index[(aa-1,bb,mm,nn)]] ) \
                                                                         + np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.eege_index[(aa,bb,mm+1,nn)]] \
                                                                                                               + self.cg[(mm-1,1)] * Sv[self.eege_index[(aa,bb,mm-1,nn)]] ) \
                                                                        )
        
            # ggee
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mes:
                        for nn in self.Mes:
                
                            # Driving
                            dSv[self.ggee_index[(aa,bb,mm,nn)]] += 1j * (param.Omet_Pi * ( self.cg[(aa,0)] * Sv[self.egee_index[(aa,bb,mm,nn)]] \
                                                                                         - self.cg[(nn,0)] * Sv[self.ggeg_index[(aa,bb,mm,nn)]] )\
                                                                         - np.conj(param.Omet_Pi) * ( self.cg[(bb,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] \
                                                                                                    - self.cg[(mm,0)] * Sv[self.ggge_index[(aa,bb,mm,nn)]] ) \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.egee_index[(aa-1,bb,mm,nn)]] \
                                                                                                      + self.cg[(aa,1)] * Sv[self.egee_index[(aa+1,bb,mm,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.ggeg_index[(aa,bb,mm,nn+1)]] \
                                                                                                      + self.cg[(nn-1,1)] * Sv[self.ggeg_index[(aa,bb,mm,nn-1)]] ) \
                                                                         - np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(bb,-1)] * Sv[self.geee_index[(aa,bb-1,mm,nn)]] \
                                                                                                               + self.cg[(bb,1)] * Sv[self.geee_index[(aa,bb+1,mm,nn)]] ) \
                                                                         + np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.ggge_index[(aa,bb,mm+1,nn)]] \
                                                                                                               + self.cg[(mm-1,1)] * Sv[self.ggge_index[(aa,bb,mm-1,nn)]] ) \
                                                                        )
        
            # ggge
            for aa in self.Mgs:
                for bb in self.Mgs:
                    for mm in self.Mgs:
                        for nn in self.Mes:
                
                            # Driving
                            dSv[self.ggge_index[(aa,bb,mm,nn)]] += 1j * (param.Omet_Pi * ( self.cg[(aa,0)] * Sv[self.egge_index[(aa,bb,mm,nn)]] \
                                                                                         + self.cg[(mm,0)] * Sv[self.ggee_index[(aa,bb,mm,nn)]] \
                                                                                         - self.cg[(nn,0)] * Sv[self.gggg_index[(aa,bb,mm,nn)]] )\
                                                                         - np.conj(param.Omet_Pi) * self.cg[(bb,0)] * Sv[self.gege_index[(aa,bb,mm,nn)]] \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.egge_index[(aa-1,bb,mm,nn)]] \
                                                                                                      + self.cg[(aa,1)] * Sv[self.egge_index[(aa+1,bb,mm,nn)]] ) \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(mm,-1)] * Sv[self.ggee_index[(aa,bb,mm-1,nn)]] \
                                                                                                      + self.cg[(mm,1)] * Sv[self.ggee_index[(aa,bb,mm+1,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.gggg_index[(aa,bb,mm,nn+1)]] \
                                                                                                      + self.cg[(nn-1,1)] * Sv[self.gggg_index[(aa,bb,mm,nn-1)]] ) \
                                                                         - np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(bb,-1)] * Sv[self.gege_index[(aa,bb-1,mm,nn)]] \
                                                                                                               + self.cg[(bb,1)] * Sv[self.gege_index[(aa,bb+1,mm,nn)]] ) \
                                                                )
        
            # geee
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mes:
                
                            # Driving
                            dSv[self.geee_index[(aa,bb,mm,nn)]] += 1j * (param.Omet_Pi * ( self.cg[(aa,0)] * Sv[self.eeee_index[(aa,bb,mm,nn)]] \
                                                                                         - self.cg[(bb,0)] * Sv[self.ggee_index[(aa,bb,mm,nn)]] \
                                                                                         - self.cg[(nn,0)] * Sv[self.geeg_index[(aa,bb,mm,nn)]] )\
                                                                         + np.conj(param.Omet_Pi) * self.cg[(mm,0)] * Sv[self.gege_index[(aa,bb,mm,nn)]] \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eeee_index[(aa-1,bb,mm,nn)]] \
                                                                                                      + self.cg[(aa,1)] * Sv[self.eeee_index[(aa+1,bb,mm,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.ggee_index[(aa,bb+1,mm,nn)]] \
                                                                                                      + self.cg[(bb-1,1)] * Sv[self.ggee_index[(aa,bb-1,mm,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.geeg_index[(aa,bb,mm,nn+1)]] \
                                                                                                      + self.cg[(nn-1,1)] * Sv[self.geeg_index[(aa,bb,mm,nn-1)]] ) \
                                                                         + np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.gege_index[(aa,bb,mm+1,nn)]] \
                                                                                                               + self.cg[(mm-1,1)] * Sv[self.gege_index[(aa,bb,mm-1,nn)]] ) \
                                                                        )
        
            # gege
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mgs:
                        for nn in self.Mes:
                
                            # Driving
                            dSv[self.gege_index[(aa,bb,mm,nn)]] += 1j * (param.Omet_Pi * ( self.cg[(aa,0)] * Sv[self.eege_index[(aa,bb,mm,nn)]] \
                                                                                         - self.cg[(bb,0)] * Sv[self.ggge_index[(aa,bb,mm,nn)]] \
                                                                                         + self.cg[(mm,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] \
                                                                                         - self.cg[(nn,0)] * Sv[self.gegg_index[(aa,bb,mm,nn)]] )\
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eege_index[(aa-1,bb,mm,nn)]] \
                                                                                                      + self.cg[(aa,1)] * Sv[self.eege_index[(aa+1,bb,mm,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.ggge_index[(aa,bb+1,mm,nn)]] \
                                                                                                      + self.cg[(bb-1,1)] * Sv[self.ggge_index[(aa,bb-1,mm,nn)]] ) \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(mm,-1)] * Sv[self.geee_index[(aa,bb,mm-1,nn)]] \
                                                                                                      + self.cg[(mm,1)] * Sv[self.geee_index[(aa,bb,mm+1,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.gegg_index[(aa,bb,mm,nn+1)]] \
                                                                                                      + self.cg[(nn-1,1)] * Sv[self.gegg_index[(aa,bb,mm,nn-1)]] ) \
                                                                        )
        
            # geeg
            for aa in self.Mgs:
                for bb in self.Mes:
                    for mm in self.Mes:
                        for nn in self.Mgs:
                
                            # Driving
                            dSv[self.geeg_index[(aa,bb,mm,nn)]] += 1j * (param.Omet_Pi * ( self.cg[(aa,0)] * Sv[self.eeeg_index[(aa,bb,mm,nn)]] \
                                                                                         - self.cg[(bb,0)] * Sv[self.ggeg_index[(aa,bb,mm,nn)]] )\
                                                                         + np.conj(param.Omet_Pi) * ( self.cg[(mm,0)] * Sv[self.gegg_index[(aa,bb,mm,nn)]] \
                                                                                                    - self.cg[(nn,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] ) \
                                                                         + param.Omet_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eeeg_index[(aa-1,bb,mm,nn)]] \
                                                                                                      + self.cg[(aa,1)] * Sv[self.eeeg_index[(aa+1,bb,mm,nn)]] ) \
                                                                         - param.Omet_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.ggeg_index[(aa,bb+1,mm,nn)]] \
                                                                                                      + self.cg[(bb-1,1)] * Sv[self.ggeg_index[(aa,bb-1,mm,nn)]] ) \
                                                                         + np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.gegg_index[(aa,bb,mm+1,nn)]] \
                                                                                                               + self.cg[(mm-1,1)] * Sv[self.gegg_index[(aa,bb,mm-1,nn)]] ) \
                                                                         - np.conj(param.Omet_Sigma)/sqrt(2) * ( self.cg[(nn,-1)] * Sv[self.geee_index[(aa,bb,mm,nn-1)]] \
                                                                                                               + self.cg[(nn,1)] * Sv[self.geee_index[(aa,bb,mm,nn+1)]] ) \
                                                                        )
                                                                        
            return dSv.reshape( (len(dSv)*len(dSv[0])) )
        
        
        
        
        # Set solver
        if param.method in ['MF','TWA']:
            ICsolver = complex_ode(MF_eqs_IC).set_integrator('dopri5',atol=param.atol,rtol=param.rtol)
        if param.method in ['cumulant','bbgky']:
            ICsolver = complex_ode(cumulant_eqs_IC).set_integrator('dopri5',atol=param.atol,rtol=param.rtol)
            
        print('atol:', ICsolver._integrator.atol)
        print('rtol:', ICsolver._integrator.rtol)
        
        ICsolver.set_initial_value(self.Sv.reshape( (len(self.Sv)*len(self.Sv[0])) ), 0)
        
        # Evolve for t=1
        if ICsolver.successful():
            ICsolver.integrate(ICsolver.t+1)
            self.Sv = ICsolver.y.reshape((self.nvariables,self.iterations))
        else: print("\nERROR/rotate_IC: Problem with ICsolver, returns unsuccessful.\n")
        
        
        
    ######
    ######  Hilbert space functions 
    ######
    def fill_single_p_rho (self,cIC=param.cIC):
        """Initialize density matrix of one atom to chosen initial condition."""
        
        def rho_IC_initialstate ():
            """ Initialize density matrix of one atom in the state specified by param.initialstate. """
            if len(param.initialstate)!=param.filling: print("\nWarning: Length of initialstate doesn't match filling.\n")
            self.rho = np.zeros((self.localhilbertsize,self.localhilbertsize))
            occupied_localstate = self.eg_to_level[ param.initialstate[0][0] ][ int(param.initialstate[0][1:]) ]
            self.rho[occupied_localstate,occupied_localstate] = 1
            
        def rho_IC_mixedstate ():
            """ Initialize density matrix of one atom in a mixed state of ground states """
            self.rho = np.zeros((self.localhilbertsize,self.localhilbertsize))
            #for mg in range(param.deg_g): self.rho[mg,mg] = param.mixed_gs_probabilities[mg]
            for mg in range(param.deg_g): self.rho[self.eg_to_level['g'][mg],self.eg_to_level['g'][mg]] = param.mixed_gs_probabilities[mg]
            for me in range(param.deg_e): self.rho[self.eg_to_level['e'][me],self.eg_to_level['e'][me]] = param.mixed_es_probabilities[me]
            print(self.rho)
        
        def rho_IC_puresupstate ():
            """ Initialize density matrix of one atom in a pure superposition state of ground levels """
            psi_i = np.zeros((self.localhilbertsize,1),dtype=complex)
            for mg in range(param.deg_g): psi_i[mg,0] = param.pure_gs_amplitudes[mg]
            for me in range(param.deg_e): psi_i[self.eg_to_level['e'][me],0] = param.pure_es_amplitudes[me]
            self.rho = psi_i @ psi_i.conj().T
            
        def rho_IC_byhand ():
            """ Initialize density matrix of one atom in some state hardcoded by hand """
            self.rho = np.zeros((self.localhilbertsize,self.localhilbertsize), dtype=complex)
            
            # |g_{1/2}><g_{1/2}| + superpos of g_{-1/2} and e_{-1/2}
            occupied_localstate = self.eg_to_level[ 'g' ][ 1 ]
            self.rho[occupied_localstate,occupied_localstate] = 0.5
            
            psi_i = np.array( [[1/sqrt(2)], [0], [1/sqrt(2)], [0]] , dtype=complex)
            self.rho += 0.5 * psi_i @ psi_i.conj().T
        
        optionsIC = { 'initialstate': rho_IC_initialstate,
                        'mixedstate': rho_IC_mixedstate,
                        'puresupstate': rho_IC_puresupstate,
                        'byhand': rho_IC_byhand }
        optionsIC[cIC]()
        
        # Check initial trace
        if abs(np.trace(self.rho)-1)>0.000000001: print("\nWARNING/choose_initial_condition: Trace of rho is initially not 1.\n")
        print("Initial trace of rho is %g."%(np.trace(self.rho).real))
        
        
    def transitionOp_local (self,n1,n2):
        """
        Returns local transition operator |n1><n2|, where n1, n2 label states in the local Hilbert space
        """
        temp = np.zeros((self.localhilbertsize,self.localhilbertsize))
        temp[n1,n2] = 1
        return temp
        #return csc_matrix( ([1],([n1],[n2])) , shape=(self.localhilbertsize,self.localhilbertsize))
    
    
    
    ######
    ######  Previous initialization functions for MF (not in use anymore)
    ######
    '''
    def IC_initialstate (self):
        """
        Initialize all sites with the atoms in the state specified by param.initialstate.
        """
        eg_manifold = param.initialstate[0][0]
        
        if eg_manifold=='g':
            mg = self.Mgs[ int(param.initialstate[0][1:]) ]
            self.Sv[ self.gg_index[(mg,mg)] ] = 1
            print(self.Sv[ self.gg_index[(mg,mg)] ])
            
        if eg_manifold=='e':
            me = self.Mes[ int(param.initialstate[0][1:]) ]
            self.Sv[ self.ee_index[(me,me)] ] = 1
            print(self.Sv[ self.ee_index[(me,me)] ])
            
            
    def IC_mixedstate(self):
        """Initialize each atom in a mixed state of ground states"""
        for mg in self.Mgs:
            self.Sv[ self.gg_index[(mg,mg)] ] = param.mixed_gs_probabilities[ self.Mgs.index(mg) ]
        
        
    def IC_puresupstate(self):
        """Initialize density matrix with each atom in a pure superposition state of ground levels"""
        for aa in self.Mgs:
            for bb in self.Mgs:
                self.Sv[ self.gg_index[(aa,bb)] ] = np.conj( param.pure_gs_amplitudes[ self.Mgs.index(aa) ] ) * param.pure_gs_amplitudes[ self.Mgs.index(bb) ]
    '''
        
    
        
        
####################################################################

###############                EOM                ##################

####################################################################


    def MF_eqs (self,t,Svtemp):
        """Computes and returns right-hand side of mean-field equations"""
        Sv = Svtemp.reshape( (self.nvariables,self.iterations) )
        dSv = np.zeros((self.nvariables,self.iterations),dtype=complex)
        
        chitilde = param.g_coh - 1j*param.g_inc/2
        chitildeN = self.Ntotal * chitilde
        
        
        Pi_minus = 0
        Sigma_minus = 0
        for mm in self.Mgs: Pi_minus += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
        for mm in self.Mgs: Sigma_minus += 0.5 * self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]] + 0.5 * self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
        Pi_plus = np.conj(Pi_minus)
        Sigma_plus = np.conj(Sigma_minus)
        
        
        oneminus1overN = 1 - 1/self.Ntotal
        oneoverN = 1/self.Ntotal
        
        
        # gg        
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.gg_index[(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.gg_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(aa,q)] * self.cg[(bb,q)] * Sv[self.ee_index[(aa+q,bb+q)]]
                    #if aa+q in self.Mes:
                    #    if bb+q in self.Mes:
                    #        temp += self.cg[(aa,q)] * self.cg[(bb,q)] * Sv[self.ee_index[(aa+q,bb+q)]]
                dSv[self.gg_index[(aa,bb)]] += param.gamma0 * temp
                
                # Driving
                dSv[self.gg_index[(aa,bb)]] += 1j * ( param.rabi_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                      - np.conj(param.rabi_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                      + param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])\
                                                      + param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])\
                                                      - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]\
                                                      - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]\
                                                    )
                
                # Cavity
                if self.Ntotal>1:
                    dSv[self.gg_index[(aa,bb)]] += 1j * chitildeN * oneminus1overN * ( Pi_minus * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                                                      + Sigma_minus * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])\
                                                                                      + Sigma_minus * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])\
                                                                                     )
                
                    dSv[self.gg_index[(aa,bb)]] += -1j * np.conj(chitildeN) * oneminus1overN * ( Pi_plus * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                                                               + Sigma_plus * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]\
                                                                                               + Sigma_plus * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]\
                                                                                               )
                
                
                #"""
                # Cavity self-interaction terms (for TWA this symmetrization is more exact)
                if param.subleading_terms == 'selfint':
                    dSv[self.gg_index[(aa,bb)]] += - 2 * chitildeN.imag * oneoverN * ( self.cg[(aa,0)] * self.cg[(bb,0)] * Sv[self.ee_index[(aa,bb)]] \
                                                                                + 0.5 * self.cg[(aa,-1)] * ( self.cg[(bb,-1)] * Sv[self.ee_index[(aa-1,bb-1)]] \
                                                                                                            + self.cg[(bb,1)] * Sv[self.ee_index[(aa-1,bb+1)]] ) \
                                                                                + 0.5 * self.cg[(aa,1)] * ( self.cg[(bb,-1)] * Sv[self.ee_index[(aa+1,bb-1)]] \
                                                                                                            + self.cg[(bb,1)] * Sv[self.ee_index[(aa+1,bb+1)]] ) \
                                                                                )
                #"""
                
                """
                # Cavity symmetrization terms
                #if param.subleading_terms == 'sub':
                dSv[self.gg_index[(aa,bb)]] += 1j * chitilde/4.0 * ( 2.0*self.cg[(aa,0)] * ( self.cg[(bb,0)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(aa,0)] * Sv[self.gg_index[(aa,bb)]] ) \
                                                                   + self.cg[(aa,-1)] * ( self.cg[(bb,-1)] * Sv[self.ee_index[(aa-1,bb-1)]] - self.cg[(aa,-1)] * Sv[self.gg_index[(aa,bb)]] \
                                                                                         + self.cg[(bb,1)] * Sv[self.ee_index[(aa-1,bb+1)]] - self.cg[(aa-2,1)] * Sv[self.gg_index[(aa-2,bb)]] )\
                                                                   + self.cg[(aa,1)] * ( self.cg[(bb,-1)] * Sv[self.ee_index[(aa+1,bb-1)]] - self.cg[(aa+2,-1)] * Sv[self.gg_index[(aa+2,bb)]] \
                                                                                         + self.cg[(bb,1)] * Sv[self.ee_index[(aa+1,bb+1)]] - self.cg[(aa,1)] * Sv[self.gg_index[(aa,bb)]] )\
                                                                 )
                                                            
                dSv[self.gg_index[(aa,bb)]] += -1j * np.conj(chitilde)/4.0 * ( 2.0*self.cg[(bb,0)] * ( self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]] ) \
                                                                           + self.cg[(bb,-1)] * ( self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb-1)]] - self.cg[(bb,-1)] * Sv[self.gg_index[(aa,bb)]] \
                                                                                                 + self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb-1)]] - self.cg[(bb-2,1)] * Sv[self.gg_index[(aa,bb-2)]] )\
                                                                           + self.cg[(bb,1)] * ( self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb+1)]] - self.cg[(bb+2,-1)] * Sv[self.gg_index[(aa,bb+2)]] \
                                                                                                 + self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb+1)]] - self.cg[(bb,1)] * Sv[self.gg_index[(aa,bb)]] )\
                                                                           )
                #"""
                
        # ee
        for aa in self.Mes:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ee_index[(aa,bb)]] += 1j * param.zeeman_e * float(aa-bb) * Sv[self.ee_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(aa-q,q)]**2 * Sv[self.ee_index[(aa,bb)]] + self.cg[(bb-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                    #if aa-q in self.Mgs: temp += self.cg[(aa-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                    #if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                            
                dSv[self.ee_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                dSv[self.ee_index[(aa,bb)]] += 1j * (- param.rabi_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                     + np.conj(param.rabi_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])\
                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]\
                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]\
                                                    )
                
                # Cavity
                if self.Ntotal>1:
                    dSv[self.ee_index[(aa,bb)]] += - 1j * chitildeN * oneminus1overN * ( Pi_minus * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]]) \
                                                                                       + Sigma_minus * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]]) \
                                                                                       + Sigma_minus * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]]) \
                                                                                       )
                
                    dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitildeN) * oneminus1overN * ( Pi_plus * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]] \
                                                                                              + Sigma_plus * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]] \
                                                                                              + Sigma_plus * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]] \
                                                                                              )
                
                
                #"""
                # Cavity self-interaction terms (for TWA this symmetrization is more exact)
                if param.subleading_terms == 'selfint':
                    dSv[self.ee_index[(aa,bb)]] += -1j * chitildeN * oneoverN * ( self.cg[(bb,0)]**2 * Sv[self.ee_index[(aa,bb)]] \
                                                                                + 0.5 * self.cg[(bb+1,-1)] * ( self.cg[(bb+1,-1)] * Sv[self.ee_index[(aa,bb)]] \
                                                                                                             + self.cg[(bb+1,1)] * Sv[self.ee_index[(aa,bb+2)]] ) \
                                                                                + 0.5 * self.cg[(bb-1,1)] * ( self.cg[(bb-1,-1)] * Sv[self.ee_index[(aa,bb-2)]] \
                                                                                                            + self.cg[(bb-1,1)] * Sv[self.ee_index[(aa,bb)]] ) \
                                                                                )
                                                                            
                    dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitildeN) * oneoverN * ( self.cg[(aa,0)]**2 * Sv[self.ee_index[(aa,bb)]] \
                                                                                        + 0.5 * self.cg[(aa+1,-1)] * ( self.cg[(aa+1,-1)] * Sv[self.ee_index[(aa,bb)]] \
                                                                                                                     + self.cg[(aa+1,1)] * Sv[self.ee_index[(aa+2,bb)]] ) \
                                                                                        + 0.5 * self.cg[(aa-1,1)] * ( self.cg[(aa-1,-1)] * Sv[self.ee_index[(aa-2,bb)]] \
                                                                                                                    + self.cg[(aa-1,1)] * Sv[self.ee_index[(aa,bb)]] ) \
                                                                                        )
                #"""
                
                """
                # Cavity symmetrization terms
                #if param.subleading_terms == 'sub':
                dSv[self.ee_index[(aa,bb)]] += -1j * chitilde/4.0 * ( 2.0*self.cg[(bb,0)] * ( self.cg[(bb,0)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(aa,0)] * Sv[self.gg_index[(aa,bb)]] ) \
                                                                   + self.cg[(bb+1,-1)] * ( self.cg[(bb+1,-1)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(aa+1,-1)] * Sv[self.gg_index[(aa+1,bb+1)]] \
                                                                                         + self.cg[(bb+1,1)] * Sv[self.ee_index[(aa,bb+2)]] - self.cg[(aa-1,1)] * Sv[self.gg_index[(aa-1,bb+1)]] )\
                                                                   + self.cg[(bb-1,1)] * ( self.cg[(bb-1,-1)] * Sv[self.ee_index[(aa,bb-2)]] - self.cg[(aa+1,-1)] * Sv[self.gg_index[(aa+1,bb-1)]] \
                                                                                         + self.cg[(bb-1,1)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(aa-1,1)] * Sv[self.gg_index[(aa-1,bb-1)]] )\
                                                                 )
                                                            
                dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitilde)/4.0 * ( 2.0*self.cg[(aa,0)] * ( self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]] ) \
                                                                           + self.cg[(aa+1,-1)] * ( self.cg[(aa+1,-1)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa+1,bb+1)]] \
                                                                                                 + self.cg[(aa+1,1)] * Sv[self.ee_index[(aa+2,bb)]] - self.cg[(bb-1,1)] * Sv[self.gg_index[(aa+1,bb-1)]] )\
                                                                           + self.cg[(aa-1,1)] * ( self.cg[(aa-1,-1)] * Sv[self.ee_index[(aa-2,bb)]] - self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa-1,bb+1)]] \
                                                                                                 + self.cg[(aa-1,1)] * Sv[self.ee_index[(aa,bb)]] - self.cg[(bb-1,1)] * Sv[self.gg_index[(aa-1,bb-1)]] )\
                                                                           )
                                                                         
                #"""                                               
        
        # ge
        for aa in self.Mgs:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.zeeman_g*float(aa) - param.zeeman_e*float(bb) + param.detuning) * Sv[self.ge_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(bb-q,q)]**2 * Sv[self.ge_index[(aa,bb)]]
                    #if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ge_index[(aa,bb)]]
                            
                dSv[self.ge_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.rabi_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]\
                                                     - param.rabi_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]\
                                                     + param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]\
                                                     + param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]\
                                                    )
                
                # Cavity
                if self.Ntotal>1:
                    dSv[self.ge_index[(aa,bb)]] += 1j * chitildeN * oneminus1overN * ( Pi_minus * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]\
                                                                                     - Pi_minus * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]\
                                                                                     + Sigma_minus * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]\
                                                                                     + Sigma_minus * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]\
                                                                                     - Sigma_minus * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]\
                                                                                     - Sigma_minus * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]\
                                                                                     )
                
                #"""
                # Cavity self-interaction terms (for TWA this symmetrization is more exact)
                if param.subleading_terms == 'selfint':
                    dSv[self.ge_index[(aa,bb)]] += -1j * chitildeN * oneoverN * ( self.cg[(bb,0)]**2 * Sv[self.ge_index[(aa,bb)]] \
                                                                                + 0.5 * self.cg[(bb+1,-1)] * ( self.cg[(bb+1,-1)] * Sv[self.ge_index[(aa,bb)]] \
                                                                                                             + self.cg[(bb+1,1)] * Sv[self.ge_index[(aa,bb+2)]] ) \
                                                                                + 0.5 * self.cg[(bb-1,1)] * ( self.cg[(bb-1,-1)] * Sv[self.ge_index[(aa,bb-2)]] \
                                                                                                            + self.cg[(bb-1,1)] * Sv[self.ge_index[(aa,bb)]] ) \
                                                                                )
                #"""
                
                """
                # Cavity symmetrization terms
                #if param.subleading_terms == 'sub':
                dSv[self.ge_index[(aa,bb)]] += -1j * chitilde/4.0 * ( 2.0 * ( self.cg[(aa,0)]**2 + self.cg[(bb,0)]**2 ) * Sv[self.ge_index[(aa,bb)]] \
                                                                   + self.cg[(aa,-1)] * ( self.cg[(aa,-1)] * Sv[self.ge_index[(aa,bb)]] + self.cg[(aa-2,1)] * Sv[self.ge_index[(aa-2,bb)]] ) \
                                                                   + self.cg[(aa,1)] * ( self.cg[(aa+2,-1)] * Sv[self.ge_index[(aa+2,bb)]] + self.cg[(aa,1)] * Sv[self.ge_index[(aa,bb)]] ) \
                                                                   + self.cg[(bb+1,-1)] * ( self.cg[(bb+1,-1)] * Sv[self.ge_index[(aa,bb)]] + self.cg[(bb+1,1)] * Sv[self.ge_index[(aa,bb+2)]] ) \
                                                                   + self.cg[(bb-1,1)] * ( self.cg[(bb-1,-1)] * Sv[self.ge_index[(aa,bb-2)]] + self.cg[(bb-1,1)] * Sv[self.ge_index[(aa,bb)]] ) \
                                                                 )
                #"""
                
        
        return dSv.reshape( (len(dSv)*len(dSv[0])) )

    
    
    def cumulant_eqs (self,t,Svtemp):
        """Computes and returns right-hand side of cumulant equations"""
        Sv = Svtemp.reshape( (self.nvariables,self.iterations) )
        dSv = np.zeros((self.nvariables,self.iterations),dtype=complex)
        
        # Compute arrays of auxiliary variables and add to Sv array
        # Note: After this step, the array Sv has size (self.nvariables+self.n_auxvariables , self.iterations),
        #       rows 0 to nvariables-1 correspond to main variables, the rest are auxiliary variables
        Sv = self.fill_auxiliary_variables(Sv)
        
        chitilde = param.g_coh - 1j*param.g_inc/2
        chitildeN = self.Ntotal * chitilde
        
        if param.method=='cumulant':
            Nm2overN = (self.Ntotal-2)/self.Ntotal
            Nm1m2overN2 = (self.Ntotal-1)*(self.Ntotal-2)/self.Ntotal**2
            Nm2overN2 = (self.Ntotal-2)/self.Ntotal**2
            oneoverN = 1/self.Ntotal            ############ CAREFUL WITH THIS WHEN USING SPONTANEOUS EMISSION
            oneoverN2 = 1/self.Ntotal**2
            
        if param.method=='bbgky':
            Nm2overN = (self.Ntotal-2)/self.Ntotal
            Nm1m2overN2 = (self.Ntotal-1)*(self.Ntotal-2)/self.Ntotal**2
            Nm2overN2 = (self.Ntotal-2)/self.Ntotal**2
            oneoverN = 1/self.Ntotal            ############ CAREFUL WITH THIS WHEN USING SPONTANEOUS EMISSION
            oneoverN2 = 1/self.Ntotal**2
            
            #Nm2overN = 1
            #Nm1m2overN2 = 1
            #Nm2overN2 = 0
            #oneoverN = 0
            #oneoverN2 = 0
            oneover2N = 1/(2*self.Ntotal)
        
        ###
        ###     1pt eqs
        ###
        
        # gg
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.gg_index[(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.gg_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(aa,q)] * self.cg[(bb,q)] * Sv[self.ee_index[(aa+q,bb+q)]]
                dSv[self.gg_index[(aa,bb)]] += param.gamma0 * temp
                
                # Driving
                dSv[self.gg_index[(aa,bb)]] += 1j * ( param.rabi_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                      - np.conj(param.rabi_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                      + param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])\
                                                      + param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])\
                                                      - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]\
                                                      - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]\
                                                    )
                
                # Cavity
                dSv[self.gg_index[(aa,bb)]] += 1j * chitildeN/2.0 * ( 2 * self.cg[(aa,0)] * Sv[self.index['egPminus'][(aa,bb)]] \
                                                                  + self.cg[(aa,-1)] * Sv[self.index['egSminus'][(aa-1,bb)]] \
                                                                  + self.cg[(aa,1)] * Sv[self.index['egSminus'][(aa+1,bb)]] \
                                                                )
                
                dSv[self.gg_index[(aa,bb)]] += -1j * np.conj(chitildeN)/2.0 * ( 2* self.cg[(bb,0)] * Sv[self.index['Pplusge'][(aa,bb)]] \
                                                                               + self.cg[(bb,-1)] * Sv[self.index['Splusge'][(aa,bb-1)]] \
                                                                               + self.cg[(bb,1)] * Sv[self.index['Splusge'][(aa,bb+1)]] \
                                                                          )
                
        # ee
        for aa in self.Mes:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ee_index[(aa,bb)]] += 1j * param.zeeman_e * float(aa-bb) * Sv[self.ee_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += (self.cg[(aa-q,q)]**2 + self.cg[(bb-q,q)]**2) * Sv[self.ee_index[(aa,bb)]]
                dSv[self.ee_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                dSv[self.ee_index[(aa,bb)]] += 1j * (- param.rabi_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                     + np.conj(param.rabi_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])\
                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]\
                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]\
                                                    )
                
                # Cavity
                dSv[self.ee_index[(aa,bb)]] += - 1j * chitildeN/2.0 * ( 2* self.cg[(bb,0)] * Sv[self.index['egPminus'][(aa,bb)]] \
                                                                       + self.cg[(bb+1,-1)] * Sv[self.index['egSminus'][(aa,bb+1)]] \
                                                                       + self.cg[(bb-1,1)] * Sv[self.index['egSminus'][(aa,bb-1)]] \
                                                                     )
                
                dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitildeN)/2.0 * ( 2* self.cg[(aa,0)] * Sv[self.index['Pplusge'][(aa,bb)]] \
                                                                          + self.cg[(aa+1,-1)] * Sv[self.index['Splusge'][(aa+1,bb)]] \
                                                                          + self.cg[(aa-1,1)] * Sv[self.index['Splusge'][(aa-1,bb)]] \
                                                                         )
        
                
        # ge
        for aa in self.Mgs:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.zeeman_g*float(aa) - param.zeeman_e*float(bb) + param.detuning) * Sv[self.ge_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(bb-q,q)]**2 * Sv[self.ge_index[(aa,bb)]]
                dSv[self.ge_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.rabi_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]\
                                                     - param.rabi_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]\
                                                     + param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]\
                                                     + param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]\
                                                    )
                
                # Cavity
                dSv[self.ge_index[(aa,bb)]] += 1j * chitildeN/2.0 * (2 * self.cg[(aa,0)] * Sv[self.index['eePminus'][(aa,bb)]] \
                                                                 - 2 * self.cg[(bb,0)] * Sv[self.index['ggPminus'][(aa,bb)]] \
                                                                 + self.cg[(aa,-1)] * Sv[self.index['eeSminus'][(aa-1,bb)]] \
                                                                 + self.cg[(aa,1)] * Sv[self.index['eeSminus'][(aa+1,bb)]] \
                                                                 - self.cg[(bb+1,-1)] * Sv[self.index['ggSminus'][(aa,bb+1)]] \
                                                                 - self.cg[(bb-1,1)] * Sv[self.index['ggSminus'][(aa,bb-1)]] \
                                                                )
                
                
        
        ###
        ###     2pt eqs
        ###        
        def three_minus ( s1, s2, ps, ab1, ab2 ):
            """
            Computes three point correlator approximating 3pt-cumulant = zero.
            Assumes that s1, s2, ps are strings, and ab1, ab2 are tuples with two entries.
            NOTE: Indexing assumes that Pi/Sigma is position 3, i.e. < s1_ab1 * s2_ab2 * Pi/Sigma >.
            """
            """
            return Sv[self.index[s1+s2][ab1+ab2]] * Sv[self.index[ps]] \
                    + Sv[self.index[s1+ps][ab1]] * Sv[self.index[s2][ab2]] \
                    + Sv[self.index[s2+ps][ab2]] * Sv[self.index[s1][ab1]] \
                    - 2 * Sv[self.index[s1][ab1]] * Sv[self.index[s2][ab2]] * Sv[self.index[ps]]
            
            #"""
            
            #"""
            ###
            ### Taking self-interactions properly into account
            ###
            a = s1[0]
            b = s1[1]
            m = s2[0]
            n = s2[1]
            
            alpha = ab1[0]
            beta = ab1[1]
            mu = ab2[0]
            nu = ab2[1]
            
            if self.Ntotal==2:
                
                if ps=='Pminus':
                    
                    return self.kron_del(b,m) * self.kron_del(beta,mu) * 0.5 * Sv[self.index[a+n+ps][(alpha,nu)]] \
                           + self.cg[(beta,0)] * self.kron_del(b,'g') * ( 0.5 * Sv[self.index[a+'e'+m+n][(alpha,beta,mu,nu)]] \
                                                                        - 0.25 * self.kron_del('e',m) * self.kron_del(beta,mu) * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        ) \
                           + self.cg[(nu,0)] * self.kron_del(n,'g') * ( 0.5 * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu)]] \
                                                                        - 0.25 * self.kron_del(b,m) * self.kron_del(beta,mu) * Sv[self.index[a+'e'][(alpha,nu)]] \
                                                                        )
                    
                if ps=='Sminus':
                    
                    return self.kron_del(b,m) * self.kron_del(beta,mu) * 0.5 * Sv[self.index[a+n+ps][(alpha,nu)]] \
                           + self.cg[(beta,-1)] * self.kron_del(b,'g') * ( 0.5 * Sv[self.index[a+'e'+m+n][(alpha,beta-1,mu,nu)]] \
                                                                        - 0.25 * self.kron_del('e',m) * self.kron_del(beta-1,mu) * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        ) \
                           + self.cg[(nu,-1)] * self.kron_del(n,'g') * ( 0.5 * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu-1)]] \
                                                                        - 0.25 * self.kron_del(b,m) * self.kron_del(beta,mu) * Sv[self.index[a+'e'][(alpha,nu-1)]] \
                                                                        ) \
                           + self.cg[(beta,1)] * self.kron_del(b,'g') * ( 0.5 * Sv[self.index[a+'e'+m+n][(alpha,beta+1,mu,nu)]] \
                                                                        - 0.25 * self.kron_del('e',m) * self.kron_del(beta+1,mu) * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        ) \
                           + self.cg[(nu,1)] * self.kron_del(n,'g') * ( 0.5 * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu+1)]] \
                                                                        - 0.25 * self.kron_del(b,m) * self.kron_del(beta,mu) * Sv[self.index[a+'e'][(alpha,nu+1)]] \
                                                                        )
                
            else:
            
                if ps=='Pminus':
                    return Nm2overN * ( Sv[self.index[s1+s2][ab1+ab2]] * Sv[self.index[ps]] \
                                        + Sv[self.index[s1+ps][ab1]] * Sv[self.index[s2][ab2]] \
                                        + Sv[self.index[s2+ps][ab2]] * Sv[self.index[s1][ab1]] \
                                      ) \
                           - 2 * Nm1m2overN2 * Sv[self.index[s1][ab1]] * Sv[self.index[s2][ab2]] * Sv[self.index[ps]] \
                           + self.kron_del(b,m) * self.kron_del(beta,mu) * ( oneoverN * Sv[self.index[a+n+ps][(alpha,nu)]] \
                                                                            - Nm2overN2 * Sv[self.index[a+n][(alpha,nu)]] * Sv[self.index[ps]] \
                                                                            ) \
                           + self.cg[(beta,0)] * self.kron_del(b,'g') * ( oneoverN * Sv[self.index[a+'e'+m+n][(alpha,beta,mu,nu)]] \
                                                                        - oneoverN2 * self.kron_del('e',m) * self.kron_del(beta,mu) * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[a+'e'][(alpha,beta)]] * Sv[self.index[m+n][(mu,nu)]] \
                                                                        ) \
                           + self.cg[(nu,0)] * self.kron_del(n,'g') * ( oneoverN * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu)]] \
                                                                        - oneoverN2 * self.kron_del(b,m) * self.kron_del(beta,mu) * Sv[self.index[a+'e'][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[m+'e'][(mu,nu)]] * Sv[self.index[a+b][(alpha,beta)]] \
                                                                        )
                
                if ps=='Sminus':
                
                    return Nm2overN * ( Sv[self.index[s1+s2][ab1+ab2]] * Sv[self.index[ps]] \
                                        + Sv[self.index[s1+ps][ab1]] * Sv[self.index[s2][ab2]] \
                                        + Sv[self.index[s2+ps][ab2]] * Sv[self.index[s1][ab1]] \
                                      ) \
                           - 2 * Nm1m2overN2 * Sv[self.index[s1][ab1]] * Sv[self.index[s2][ab2]] * Sv[self.index[ps]] \
                           + self.kron_del(b,m) * self.kron_del(beta,mu) * ( oneoverN * Sv[self.index[a+n+ps][(alpha,nu)]] \
                                                                            - Nm2overN2 * Sv[self.index[a+n][(alpha,nu)]] * Sv[self.index[ps]] \
                                                                            ) \
                           + self.cg[(beta,-1)] * self.kron_del(b,'g') * ( oneoverN * Sv[self.index[a+'e'+m+n][(alpha,beta-1,mu,nu)]] \
                                                                        - oneoverN2 * self.kron_del('e',m) * self.kron_del(beta-1,mu) * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[a+'e'][(alpha,beta-1)]] * Sv[self.index[m+n][(mu,nu)]] \
                                                                        ) \
                           + self.cg[(nu,-1)] * self.kron_del(n,'g') * ( oneoverN * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu-1)]] \
                                                                        - oneoverN2 * self.kron_del(b,m) * self.kron_del(beta,mu) * Sv[self.index[a+'e'][(alpha,nu-1)]] \
                                                                        - Nm2overN2 * Sv[self.index[m+'e'][(mu,nu-1)]] * Sv[self.index[a+b][(alpha,beta)]] \
                                                                        ) \
                           + self.cg[(beta,1)] * self.kron_del(b,'g') * ( oneoverN * Sv[self.index[a+'e'+m+n][(alpha,beta+1,mu,nu)]] \
                                                                        - oneoverN2 * self.kron_del('e',m) * self.kron_del(beta+1,mu) * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[a+'e'][(alpha,beta+1)]] * Sv[self.index[m+n][(mu,nu)]] \
                                                                        ) \
                           + self.cg[(nu,1)] * self.kron_del(n,'g') * ( oneoverN * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu+1)]] \
                                                                        - oneoverN2 * self.kron_del(b,m) * self.kron_del(beta,mu) * Sv[self.index[a+'e'][(alpha,nu+1)]] \
                                                                        - Nm2overN2 * Sv[self.index[m+'e'][(mu,nu+1)]] * Sv[self.index[a+b][(alpha,beta)]] \
                                                                        )
            #"""                                                            
        
        ## three_minus version with kroenecker_delta as dictionary. A bit faster, but not much.
        '''
        self.kdel = {}
        self.kdel[('g','g')] = 1
        self.kdel[('e','e')] = 1
        for mg in self.Mgs: self.kdel[(mg,mg)] = 1
        for me in self.Mes: self.kdel[(me,me)] = 1
        self.kdel = defaultdict( lambda: 0, self.kdel )     # Set all other Clebsch-Gordan coefficients to zero

        def three_minus ( s1, s2, ps, ab1, ab2 ):
            """
            Computes three point correlator approximating 3pt-cumulant = zero.
            Assumes that s1, s2, ps are strings, and ab1, ab2 are tuples with two entries.
            NOTE: Indexing assumes that Pi/Sigma is position 3, i.e. < s1_ab1 * s2_ab2 * Pi/Sigma >.
            """
            #return Sv[self.index[s1+s2][ab1+ab2]] * Sv[self.index[ps]] \
            #        + Sv[self.index[s1+ps][ab1]] * Sv[self.index[s2][ab2]] \
            #        + Sv[self.index[s2+ps][ab2]] * Sv[self.index[s1][ab1]] \
            #        - 2 * Sv[self.index[s1][ab1]] * Sv[self.index[s2][ab2]] * Sv[self.index[ps]]
            
            #"""
            
            ###
            ### Taking self-interactions properly into account
            ###
            a = s1[0]
            b = s1[1]
            m = s2[0]
            n = s2[1]
            
            alpha = ab1[0]
            beta = ab1[1]
            mu = ab2[0]
            nu = ab2[1]
            
            if self.Ntotal==2:
                
                if ps=='Pminus':
                    
                    return self.kdel[(b,m)] * self.kdel[(beta,mu)] * 0.5 * Sv[self.index[a+n+ps][(alpha,nu)]] \
                           + self.cg[(beta,0)] * self.kdel[(b,'g')] * ( 0.5 * Sv[self.index[a+'e'+m+n][(alpha,beta,mu,nu)]] \
                                                                        - 0.25 * self.kdel[('e',m)] * self.kdel[(beta,mu)] * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        ) \
                           + self.cg[(nu,0)] * self.kdel[(n,'g')] * ( 0.5 * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu)]] \
                                                                        - 0.25 * self.kdel[(b,m)] * self.kdel[(beta,mu)] * Sv[self.index[a+'e'][(alpha,nu)]] \
                                                                        )
                    
                if ps=='Sminus':
                    
                    return self.kdel[(b,m)] * self.kdel[(beta,mu)] * 0.5 * Sv[self.index[a+n+ps][(alpha,nu)]] \
                           + self.cg[(beta,-1)] * self.kdel[(b,'g')] * ( 0.5 * Sv[self.index[a+'e'+m+n][(alpha,beta-1,mu,nu)]] \
                                                                        - 0.25 * self.kdel[('e',m)] * self.kdel[(beta-1,mu)] * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        ) \
                           + self.cg[(nu,-1)] * self.kdel[(n,'g')] * ( 0.5 * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu-1)]] \
                                                                        - 0.25 * self.kdel[(b,m)] * self.kdel[(beta,mu)] * Sv[self.index[a+'e'][(alpha,nu-1)]] \
                                                                        ) \
                           + self.cg[(beta,1)] * self.kdel[(b,'g')] * ( 0.5 * Sv[self.index[a+'e'+m+n][(alpha,beta+1,mu,nu)]] \
                                                                        - 0.25 * self.kdel[('e',m)] * self.kdel[(beta+1,mu)] * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        ) \
                           + self.cg[(nu,1)] * self.kdel[(n,'g')] * ( 0.5 * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu+1)]] \
                                                                        - 0.25 * self.kdel[(b,m)] * self.kdel[(beta,mu)] * Sv[self.index[a+'e'][(alpha,nu+1)]] \
                                                                        )
                
            else:
            
                if ps=='Pminus':
                    return Nm2overN * ( Sv[self.index[s1+s2][ab1+ab2]] * Sv[self.index[ps]] \
                                        + Sv[self.index[s1+ps][ab1]] * Sv[self.index[s2][ab2]] \
                                        + Sv[self.index[s2+ps][ab2]] * Sv[self.index[s1][ab1]] \
                                      ) \
                           - 2 * Nm1m2overN2 * Sv[self.index[s1][ab1]] * Sv[self.index[s2][ab2]] * Sv[self.index[ps]] \
                           + self.kdel[(b,m)] * self.kdel[(beta,mu)] * ( oneoverN * Sv[self.index[a+n+ps][(alpha,nu)]] \
                                                                            - Nm2overN2 * Sv[self.index[a+n][(alpha,nu)]] * Sv[self.index[ps]] \
                                                                            ) \
                           + self.cg[(beta,0)] * self.kdel[(b,'g')] * ( oneoverN * Sv[self.index[a+'e'+m+n][(alpha,beta,mu,nu)]] \
                                                                        - oneoverN2 * self.kdel[('e',m)] * self.kdel[(beta,mu)] * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[a+'e'][(alpha,beta)]] * Sv[self.index[m+n][(mu,nu)]] \
                                                                        ) \
                           + self.cg[(nu,0)] * self.kdel[(n,'g')] * ( oneoverN * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu)]] \
                                                                        - oneoverN2 * self.kdel[(b,m)] * self.kdel[(beta,mu)] * Sv[self.index[a+'e'][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[m+'e'][(mu,nu)]] * Sv[self.index[a+b][(alpha,beta)]] \
                                                                        )
                
                if ps=='Sminus':
                
                    return Nm2overN * ( Sv[self.index[s1+s2][ab1+ab2]] * Sv[self.index[ps]] \
                                        + Sv[self.index[s1+ps][ab1]] * Sv[self.index[s2][ab2]] \
                                        + Sv[self.index[s2+ps][ab2]] * Sv[self.index[s1][ab1]] \
                                      ) \
                           - 2 * Nm1m2overN2 * Sv[self.index[s1][ab1]] * Sv[self.index[s2][ab2]] * Sv[self.index[ps]] \
                           + self.kdel[(b,m)] * self.kdel[(beta,mu)] * ( oneoverN * Sv[self.index[a+n+ps][(alpha,nu)]] \
                                                                            - Nm2overN2 * Sv[self.index[a+n][(alpha,nu)]] * Sv[self.index[ps]] \
                                                                            ) \
                           + self.cg[(beta,-1)] * self.kdel[(b,'g')] * ( oneoverN * Sv[self.index[a+'e'+m+n][(alpha,beta-1,mu,nu)]] \
                                                                        - oneoverN2 * self.kdel[('e',m)] * self.kdel[(beta-1,mu)] * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[a+'e'][(alpha,beta-1)]] * Sv[self.index[m+n][(mu,nu)]] \
                                                                        ) \
                           + self.cg[(nu,-1)] * self.kdel[(n,'g')] * ( oneoverN * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu-1)]] \
                                                                        - oneoverN2 * self.kdel[(b,m)] * self.kdel[(beta,mu)] * Sv[self.index[a+'e'][(alpha,nu-1)]] \
                                                                        - Nm2overN2 * Sv[self.index[m+'e'][(mu,nu-1)]] * Sv[self.index[a+b][(alpha,beta)]] \
                                                                        ) \
                           + self.cg[(beta,1)] * self.kdel[(b,'g')] * ( oneoverN * Sv[self.index[a+'e'+m+n][(alpha,beta+1,mu,nu)]] \
                                                                        - oneoverN2 * self.kdel[('e',m)] * self.kdel[(beta+1,mu)] * Sv[self.index[a+n][(alpha,nu)]] \
                                                                        - Nm2overN2 * Sv[self.index[a+'e'][(alpha,beta+1)]] * Sv[self.index[m+n][(mu,nu)]] \
                                                                        ) \
                           + self.cg[(nu,1)] * self.kdel[(n,'g')] * ( oneoverN * Sv[self.index[a+b+m+'e'][(alpha,beta,mu,nu+1)]] \
                                                                        - oneoverN2 * self.kdel[(b,m)] * self.kdel[(beta,mu)] * Sv[self.index[a+'e'][(alpha,nu+1)]] \
                                                                        - Nm2overN2 * Sv[self.index[m+'e'][(mu,nu+1)]] * Sv[self.index[a+b][(alpha,beta)]] \
                                                                        )
        '''
        
        def three_plus ( ps, s1, s2, ab1, ab2 ):
            """
            Computes three point correlator approximating 3pt-cumulant = zero.
            Assumes that s1, s2, ps are strings, and ab1, ab2 are tuples with two entries.
            NOTE: Indexing assumes that Pi/Sigma is position 3, i.e. < Pi/Sigma * s1_ab1 * s2_ab2 >.
            """
            #return Sv[self.index[s1+s2][ab1+ab2]] * Sv[self.index[ps]] \
            #        + Sv[self.index[ps+s1][ab1]] * Sv[self.index[s2][ab2]] \
            #        + Sv[self.index[ps+s2][ab2]] * Sv[self.index[s1][ab1]] \
            #        - 2 * Sv[self.index[s1][ab1]] * Sv[self.index[s2][ab2]] * Sv[self.index[ps]]
            
            if ps=='Pplus':
                return np.conj(three_minus( s2[1]+s2[0] , s1[1]+s1[0] , 'Pminus', (ab2[1],ab2[0]) , (ab1[1],ab1[0]) ))
            if ps=='Splus':
                return np.conj(three_minus( s2[1]+s2[0] , s1[1]+s1[0] , 'Sminus', (ab2[1],ab2[0]) , (ab1[1],ab1[0]) ))
        
        
        # gggg
        for aa in self.Mgs:
            for bb in self.Mgs:
                for mm in self.Mgs:
                    for nn in self.Mgs:
                        
                        # B-field and energies
                        dSv[self.gggg_index[(aa,bb,mm,nn)]] += 1j * param.zeeman_g * float(aa-bb+mm-nn) * Sv[self.gggg_index[(aa,bb,mm,nn)]]
                
                        # Spontaneous emission
                        temp = 0
                        for q in self.Qs:   temp += self.cg[(nn,q)] * self.cg[(aa,q)] * oneoverN * self.kron_del(bb,mm) * Sv[self.ee_index[(aa+q,nn+q)]] \
                                                    + self.cg[(bb,q)] * self.cg[(aa,q)] * Sv[self.eegg_index[(aa+q,bb+q,mm,nn)]] \
                                                    + self.cg[(nn,q)] * self.cg[(mm,q)] * Sv[self.ggee_index[(aa,bb,mm+q,nn+q)]]
                        dSv[self.gggg_index[(aa,bb,mm,nn)]] += param.gamma0 * temp
                
                        # Driving
                        dSv[self.gggg_index[(aa,bb,mm,nn)]] += 1j * (param.rabi_Pi * ( self.cg[(aa,0)] * Sv[self.eggg_index[(aa,bb,mm,nn)]] \
                                                                                     + self.cg[(mm,0)] * Sv[self.ggeg_index[(aa,bb,mm,nn)]] )\
                                                                     - np.conj(param.rabi_Pi) * ( self.cg[(bb,0)] * Sv[self.gegg_index[(aa,bb,mm,nn)]] \
                                                                                                + self.cg[(nn,0)] * Sv[self.ggge_index[(aa,bb,mm,nn)]] ) \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eggg_index[(aa-1,bb,mm,nn)]] \
                                                                                                  + self.cg[(aa,1)] * Sv[self.eggg_index[(aa+1,bb,mm,nn)]] ) \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(mm,-1)] * Sv[self.ggeg_index[(aa,bb,mm-1,nn)]] \
                                                                                                  + self.cg[(mm,1)] * Sv[self.ggeg_index[(aa,bb,mm+1,nn)]] ) \
                                                                     - np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(bb,-1)] * Sv[self.gegg_index[(aa,bb-1,mm,nn)]] \
                                                                                                           + self.cg[(bb,1)] * Sv[self.gegg_index[(aa,bb+1,mm,nn)]] ) \
                                                                     - np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(nn,-1)] * Sv[self.ggge_index[(aa,bb,mm,nn-1)]] \
                                                                                                           + self.cg[(nn,1)] * Sv[self.ggge_index[(aa,bb,mm,nn+1)]] ) \
                                                            )
                
                        # Cavity
                        dSv[self.gggg_index[(aa,bb,mm,nn)]] += 1j * chitildeN/2.0 * (2 * self.cg[(aa,0)] * three_minus('eg','gg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + 2 * self.cg[(mm,0)] * three_minus('gg','eg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,-1)] * three_minus('eg','gg','Sminus',(aa-1,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,1)] * three_minus('eg','gg','Sminus',(aa+1,bb),(mm,nn)) \
                                                                                    + self.cg[(mm,-1)] * three_minus('gg','eg','Sminus',(aa,bb),(mm-1,nn)) \
                                                                                    + self.cg[(mm,1)] * three_minus('gg','eg','Sminus',(aa,bb),(mm+1,nn)) \
                                                                                    )\
                                                             - 1j * np.conj(chitildeN)/2.0 * ( 2 * self.cg[(bb,0)] * three_plus('Pplus','ge','gg',(aa,bb),(mm,nn)) \
                                                                                             + 2 * self.cg[(nn,0)] * three_plus('Pplus','gg','ge',(aa,bb),(mm,nn)) \
                                                                                             + self.cg[(bb,-1)] * three_plus('Splus','ge','gg',(aa,bb-1),(mm,nn)) \
                                                                                             + self.cg[(bb,1)] * three_plus('Splus','ge','gg',(aa,bb+1),(mm,nn)) \
                                                                                             + self.cg[(nn,-1)] * three_plus('Splus','gg','ge',(aa,bb),(mm,nn-1)) \
                                                                                             + self.cg[(nn,1)] * three_plus('Splus','gg','ge',(aa,bb),(mm,nn+1)) \
                                                                                             )
                        """
                        # Symmetrize BBGKY
                        if param.method=='bbgky':
                            dSv[self.gggg_index[(aa,bb,mm,nn)]] += oneover2N * self.kron_del(nn,aa) * dSv[self.gg_index[(mm,bb)]] \
                                                                    - oneover2N * self.kron_del(bb,mm) * dSv[self.gg_index[(aa,nn)]]
                        #"""
                        
                        
        
            
        # eeee
        for aa in self.Mes:
            for bb in self.Mes:
                for mm in self.Mes:
                    for nn in self.Mes:
                        
                        # B-field and energies
                        dSv[self.eeee_index[(aa,bb,mm,nn)]] += 1j * param.zeeman_e * float(aa-bb+mm-nn) * Sv[self.eeee_index[(aa,bb,mm,nn)]]
                
                        # Spontaneous emission
                        temp = 0
                        for q in self.Qs:   temp += 2 * self.cg[(mm-q,q)] * self.cg[(bb-q,q)] * oneoverN * self.kron_del(bb,mm) * Sv[self.ee_index[(aa,nn)]] \
                                                    - ( self.cg[(bb-q,q)]**2 + self.cg[(nn-q,q)]**2 + self.cg[(aa-q,q)]**2 + self.cg[(mm-q,q)]**2 ) * Sv[self.eeee_index[(aa,bb,mm,nn)]]
                        dSv[self.eeee_index[(aa,bb,mm,nn)]] += param.gamma0/2 * temp
                
                        # Driving
                        dSv[self.eeee_index[(aa,bb,mm,nn)]] += 1j * ( - param.rabi_Pi * ( self.cg[(bb,0)] * Sv[self.egee_index[(aa,bb,mm,nn)]] \
                                                                                        + self.cg[(nn,0)] * Sv[self.eeeg_index[(aa,bb,mm,nn)]] )\
                                                                     + np.conj(param.rabi_Pi) * ( self.cg[(aa,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] \
                                                                                                + self.cg[(mm,0)] * Sv[self.eege_index[(aa,bb,mm,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.egee_index[(aa,bb+1,mm,nn)]] \
                                                                                                  + self.cg[(bb-1,1)] * Sv[self.egee_index[(aa,bb-1,mm,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.eeeg_index[(aa,bb,mm,nn+1)]] \
                                                                                                  + self.cg[(nn-1,1)] * Sv[self.eeeg_index[(aa,bb,mm,nn-1)]] ) \
                                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(aa+1,-1)] * Sv[self.geee_index[(aa+1,bb,mm,nn)]] \
                                                                                                           + self.cg[(aa-1,1)] * Sv[self.geee_index[(aa-1,bb,mm,nn)]] ) \
                                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.eege_index[(aa,bb,mm+1,nn)]] \
                                                                                                           + self.cg[(mm-1,1)] * Sv[self.eege_index[(aa,bb,mm-1,nn)]] ) \
                                                                    )
                
                        # Cavity
                        dSv[self.eeee_index[(aa,bb,mm,nn)]] += - 1j * chitildeN/2.0 * (2 * self.cg[(bb,0)] * three_minus('eg','ee','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + 2 * self.cg[(nn,0)] * three_minus('ee','eg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + self.cg[(bb+1,-1)] * three_minus('eg','ee','Sminus',(aa,bb+1),(mm,nn)) \
                                                                                    + self.cg[(bb-1,1)] * three_minus('eg','ee','Sminus',(aa,bb-1),(mm,nn)) \
                                                                                    + self.cg[(nn+1,-1)] * three_minus('ee','eg','Sminus',(aa,bb),(mm,nn+1)) \
                                                                                    + self.cg[(nn-1,1)] * three_minus('ee','eg','Sminus',(aa,bb),(mm,nn-1)) \
                                                                                    )\
                                                             + 1j * np.conj(chitildeN)/2.0 * ( 2 * self.cg[(aa,0)] * three_plus('Pplus','ge','ee',(aa,bb),(mm,nn)) \
                                                                                             + 2 * self.cg[(mm,0)] * three_plus('Pplus','ee','ge',(aa,bb),(mm,nn)) \
                                                                                             + self.cg[(aa+1,-1)] * three_plus('Splus','ge','ee',(aa+1,bb),(mm,nn)) \
                                                                                             + self.cg[(aa-1,1)] * three_plus('Splus','ge','ee',(aa-1,bb),(mm,nn)) \
                                                                                             + self.cg[(mm+1,-1)] * three_plus('Splus','ee','ge',(aa,bb),(mm+1,nn)) \
                                                                                             + self.cg[(mm-1,1)] * three_plus('Splus','ee','ge',(aa,bb),(mm-1,nn)) \
                                                                                             )
                                                                                             
                        """
                        # Symmetrize BBGKY
                        if param.method=='bbgky':
                            dSv[self.eeee_index[(aa,bb,mm,nn)]] += oneover2N * self.kron_del(nn,aa) * dSv[self.ee_index[(mm,bb)]] \
                                                                    - oneover2N * self.kron_del(bb,mm) * dSv[self.ee_index[(aa,nn)]]
                        #"""
        
        # ggee
        for aa in self.Mgs:
            for bb in self.Mgs:
                for mm in self.Mes:
                    for nn in self.Mes:
                        
                        # B-field and energies
                        dSv[self.ggee_index[(aa,bb,mm,nn)]] += 1j * ( param.zeeman_g * float(aa-bb) + param.zeeman_e * float(mm-nn) ) * Sv[self.ggee_index[(aa,bb,mm,nn)]]
                
                        # Spontaneous emission
                        temp = 0
                        for q in self.Qs:   temp += - 2 * self.cg[(mm-q,q)] * self.cg[(aa,q)] * oneoverN * self.kron_del(bb,mm-q) * Sv[self.ee_index[(aa+q,nn)]] \
                                                    + 2 * self.cg[(bb,q)] * self.cg[(aa,q)] * Sv[self.eeee_index[(aa+q,bb+q,mm,nn)]] \
                                                    - (self.cg[(nn-q,q)]**2 + self.cg[(mm-q,q)]**2) * Sv[self.ggee_index[(aa,bb,mm,nn)]]
                        dSv[self.ggee_index[(aa,bb,mm,nn)]] += param.gamma0/2 * temp
                
                        # Driving
                        dSv[self.ggee_index[(aa,bb,mm,nn)]] += 1j * (param.rabi_Pi * ( self.cg[(aa,0)] * Sv[self.egee_index[(aa,bb,mm,nn)]] \
                                                                                     - self.cg[(nn,0)] * Sv[self.ggeg_index[(aa,bb,mm,nn)]] )\
                                                                     - np.conj(param.rabi_Pi) * ( self.cg[(bb,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] \
                                                                                                - self.cg[(mm,0)] * Sv[self.ggge_index[(aa,bb,mm,nn)]] ) \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.egee_index[(aa-1,bb,mm,nn)]] \
                                                                                                  + self.cg[(aa,1)] * Sv[self.egee_index[(aa+1,bb,mm,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.ggeg_index[(aa,bb,mm,nn+1)]] \
                                                                                                  + self.cg[(nn-1,1)] * Sv[self.ggeg_index[(aa,bb,mm,nn-1)]] ) \
                                                                     - np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(bb,-1)] * Sv[self.geee_index[(aa,bb-1,mm,nn)]] \
                                                                                                           + self.cg[(bb,1)] * Sv[self.geee_index[(aa,bb+1,mm,nn)]] ) \
                                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.ggge_index[(aa,bb,mm+1,nn)]] \
                                                                                                           + self.cg[(mm-1,1)] * Sv[self.ggge_index[(aa,bb,mm-1,nn)]] ) \
                                                                    )
                
                        # Cavity
                        dSv[self.ggee_index[(aa,bb,mm,nn)]] += 1j * chitildeN/2.0 * (2 * self.cg[(aa,0)] * three_minus('eg','ee','Pminus',(aa,bb),(mm,nn)) \
                                                                                    - 2 * self.cg[(nn,0)] * three_minus('gg','eg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,-1)] * three_minus('eg','ee','Sminus',(aa-1,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,1)] * three_minus('eg','ee','Sminus',(aa+1,bb),(mm,nn)) \
                                                                                    - self.cg[(nn+1,-1)] * three_minus('gg','eg','Sminus',(aa,bb),(mm,nn+1)) \
                                                                                    - self.cg[(nn-1,1)] * three_minus('gg','eg','Sminus',(aa,bb),(mm,nn-1)) \
                                                                                    )\
                                                             + 1j * np.conj(chitildeN)/2.0 * ( - 2 * self.cg[(bb,0)] * three_plus('Pplus','ge','ee',(aa,bb),(mm,nn)) \
                                                                                             + 2 * self.cg[(mm,0)] * three_plus('Pplus','gg','ge',(aa,bb),(mm,nn)) \
                                                                                             - self.cg[(bb,-1)] * three_plus('Splus','ge','ee',(aa,bb-1),(mm,nn)) \
                                                                                             - self.cg[(bb,1)] * three_plus('Splus','ge','ee',(aa,bb+1),(mm,nn)) \
                                                                                             + self.cg[(mm+1,-1)] * three_plus('Splus','gg','ge',(aa,bb),(mm+1,nn)) \
                                                                                             + self.cg[(mm-1,1)] * three_plus('Splus','gg','ge',(aa,bb),(mm-1,nn)) \
                                                                                             )
        
        # ggge
        for aa in self.Mgs:
            for bb in self.Mgs:
                for mm in self.Mgs:
                    for nn in self.Mes:
                        
                        # B-field and energies
                        dSv[self.ggge_index[(aa,bb,mm,nn)]] += 1j * ( param.zeeman_g * float(aa-bb+mm) - param.zeeman_e * float(nn) + param.detuning ) * Sv[self.ggge_index[(aa,bb,mm,nn)]]
                
                        # Spontaneous emission
                        temp = 0
                        for q in self.Qs:   temp += 2 * self.cg[(bb,q)] * self.cg[(aa,q)] * Sv[self.eege_index[(aa+q,bb+q,mm,nn)]] \
                                                    - self.cg[(nn-q,q)]**2 * Sv[self.ggge_index[(aa,bb,mm,nn)]]
                        dSv[self.ggge_index[(aa,bb,mm,nn)]] += param.gamma0/2 * temp
                
                        # Driving
                        dSv[self.ggge_index[(aa,bb,mm,nn)]] += 1j * (param.rabi_Pi * ( self.cg[(aa,0)] * Sv[self.egge_index[(aa,bb,mm,nn)]] \
                                                                                     + self.cg[(mm,0)] * Sv[self.ggee_index[(aa,bb,mm,nn)]] \
                                                                                     - self.cg[(nn,0)] * Sv[self.gggg_index[(aa,bb,mm,nn)]] )\
                                                                     - np.conj(param.rabi_Pi) * self.cg[(bb,0)] * Sv[self.gege_index[(aa,bb,mm,nn)]] \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.egge_index[(aa-1,bb,mm,nn)]] \
                                                                                                  + self.cg[(aa,1)] * Sv[self.egge_index[(aa+1,bb,mm,nn)]] ) \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(mm,-1)] * Sv[self.ggee_index[(aa,bb,mm-1,nn)]] \
                                                                                                  + self.cg[(mm,1)] * Sv[self.ggee_index[(aa,bb,mm+1,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.gggg_index[(aa,bb,mm,nn+1)]] \
                                                                                                  + self.cg[(nn-1,1)] * Sv[self.gggg_index[(aa,bb,mm,nn-1)]] ) \
                                                                     - np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(bb,-1)] * Sv[self.gege_index[(aa,bb-1,mm,nn)]] \
                                                                                                           + self.cg[(bb,1)] * Sv[self.gege_index[(aa,bb+1,mm,nn)]] ) \
                                                            )
                
                        # Cavity
                        dSv[self.ggge_index[(aa,bb,mm,nn)]] += 1j * chitildeN/2.0 * (2 * self.cg[(aa,0)] * three_minus('eg','ge','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + 2 * self.cg[(mm,0)] * three_minus('gg','ee','Pminus',(aa,bb),(mm,nn)) \
                                                                                    - 2 * self.cg[(nn,0)] * three_minus('gg','gg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,-1)] * three_minus('eg','ge','Sminus',(aa-1,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,1)] * three_minus('eg','ge','Sminus',(aa+1,bb),(mm,nn)) \
                                                                                    + self.cg[(mm,-1)] * three_minus('gg','ee','Sminus',(aa,bb),(mm-1,nn)) \
                                                                                    + self.cg[(mm,1)] * three_minus('gg','ee','Sminus',(aa,bb),(mm+1,nn)) \
                                                                                    - self.cg[(nn+1,-1)] * three_minus('gg','gg','Sminus',(aa,bb),(mm,nn+1)) \
                                                                                    - self.cg[(nn-1,1)] * three_minus('gg','gg','Sminus',(aa,bb),(mm,nn-1)) \
                                                                                    )\
                                                             - 1j * np.conj(chitildeN)/2.0 * ( 2 * self.cg[(bb,0)] * three_plus('Pplus','ge','ge',(aa,bb),(mm,nn)) \
                                                                                             + self.cg[(bb,-1)] * three_plus('Splus','ge','ge',(aa,bb-1),(mm,nn)) \
                                                                                             + self.cg[(bb,1)] * three_plus('Splus','ge','ge',(aa,bb+1),(mm,nn)) \
                                                                                             )
                                                                                             
                        """
                        # Symmetrize BBGKY
                        if param.method=='bbgky':
                            dSv[self.ggge_index[(aa,bb,mm,nn)]] += - oneover2N * self.kron_del(bb,mm) * dSv[self.ge_index[(aa,nn)]]
                        #"""
        
        # geee
        for aa in self.Mgs:
            for bb in self.Mes:
                for mm in self.Mes:
                    for nn in self.Mes:
                        
                        # B-field and energies
                        dSv[self.geee_index[(aa,bb,mm,nn)]] += 1j * ( param.zeeman_g * float(aa) + param.zeeman_e * float(-bb+mm-nn) + param.detuning ) * Sv[self.geee_index[(aa,bb,mm,nn)]]
                
                        # Spontaneous emission
                        temp = 0
                        for q in self.Qs:   temp += 2 * self.cg[(mm-q,q)] * self.cg[(bb-q,q)] * oneoverN * self.kron_del(bb,mm) * Sv[self.ge_index[(aa,nn)]] \
                                                    - (self.cg[(bb-q,q)]**2 + self.cg[(nn-q,q)]**2 + self.cg[(mm-q,q)]**2) * Sv[self.geee_index[(aa,bb,mm,nn)]]
                        dSv[self.geee_index[(aa,bb,mm,nn)]] += param.gamma0/2 * temp
                
                        # Driving
                        dSv[self.geee_index[(aa,bb,mm,nn)]] += 1j * (param.rabi_Pi * ( self.cg[(aa,0)] * Sv[self.eeee_index[(aa,bb,mm,nn)]] \
                                                                                     - self.cg[(bb,0)] * Sv[self.ggee_index[(aa,bb,mm,nn)]] \
                                                                                     - self.cg[(nn,0)] * Sv[self.geeg_index[(aa,bb,mm,nn)]] )\
                                                                     + np.conj(param.rabi_Pi) * self.cg[(mm,0)] * Sv[self.gege_index[(aa,bb,mm,nn)]] \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eeee_index[(aa-1,bb,mm,nn)]] \
                                                                                                  + self.cg[(aa,1)] * Sv[self.eeee_index[(aa+1,bb,mm,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.ggee_index[(aa,bb+1,mm,nn)]] \
                                                                                                  + self.cg[(bb-1,1)] * Sv[self.ggee_index[(aa,bb-1,mm,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.geeg_index[(aa,bb,mm,nn+1)]] \
                                                                                                  + self.cg[(nn-1,1)] * Sv[self.geeg_index[(aa,bb,mm,nn-1)]] ) \
                                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.gege_index[(aa,bb,mm+1,nn)]] \
                                                                                                           + self.cg[(mm-1,1)] * Sv[self.gege_index[(aa,bb,mm-1,nn)]] ) \
                                                                    )
                
                        # Cavity
                        dSv[self.geee_index[(aa,bb,mm,nn)]] += 1j * chitildeN/2.0 * (2 * self.cg[(aa,0)] * three_minus('ee','ee','Pminus',(aa,bb),(mm,nn)) \
                                                                                    - 2 * self.cg[(bb,0)] * three_minus('gg','ee','Pminus',(aa,bb),(mm,nn)) \
                                                                                    - 2 * self.cg[(nn,0)] * three_minus('ge','eg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,-1)] * three_minus('ee','ee','Sminus',(aa-1,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,1)] * three_minus('ee','ee','Sminus',(aa+1,bb),(mm,nn)) \
                                                                                    - self.cg[(bb+1,-1)] * three_minus('gg','ee','Sminus',(aa,bb+1),(mm,nn)) \
                                                                                    - self.cg[(bb-1,1)] * three_minus('gg','ee','Sminus',(aa,bb-1),(mm,nn)) \
                                                                                    - self.cg[(nn+1,-1)] * three_minus('ge','eg','Sminus',(aa,bb),(mm,nn+1)) \
                                                                                    - self.cg[(nn-1,1)] * three_minus('ge','eg','Sminus',(aa,bb),(mm,nn-1)) \
                                                                                    )\
                                                             + 1j * np.conj(chitildeN)/2.0 * ( 2 * self.cg[(mm,0)] * three_plus('Pplus','ge','ge',(aa,bb),(mm,nn)) \
                                                                                             + self.cg[(mm+1,-1)] * three_plus('Splus','ge','ge',(aa,bb),(mm+1,nn)) \
                                                                                             + self.cg[(mm-1,1)] * three_plus('Splus','ge','ge',(aa,bb),(mm-1,nn)) \
                                                                                             )
                                                                                             
                        """
                        # Symmetrize BBGKY
                        if param.method=='bbgky':
                            dSv[self.geee_index[(aa,bb,mm,nn)]] += - oneover2N * self.kron_del(bb,mm) * dSv[self.ge_index[(aa,nn)]]
                        #"""
        
        # gege
        for aa in self.Mgs:
            for bb in self.Mes:
                for mm in self.Mgs:
                    for nn in self.Mes:
                        
                        # B-field and energies
                        dSv[self.gege_index[(aa,bb,mm,nn)]] += 1j * ( param.zeeman_g * float(aa+mm) - param.zeeman_e * float(bb+nn) + 2*param.detuning ) * Sv[self.gege_index[(aa,bb,mm,nn)]]
                
                        # Spontaneous emission
                        temp = 0
                        for q in self.Qs:   temp += (self.cg[(bb-q,q)]**2 + self.cg[(nn-q,q)]**2) * Sv[self.gege_index[(aa,bb,mm,nn)]]
                        dSv[self.gege_index[(aa,bb,mm,nn)]] += - param.gamma0/2 * temp
                
                        # Driving
                        dSv[self.gege_index[(aa,bb,mm,nn)]] += 1j * (param.rabi_Pi * ( self.cg[(aa,0)] * Sv[self.eege_index[(aa,bb,mm,nn)]] \
                                                                                     - self.cg[(bb,0)] * Sv[self.ggge_index[(aa,bb,mm,nn)]] \
                                                                                     + self.cg[(mm,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] \
                                                                                     - self.cg[(nn,0)] * Sv[self.gegg_index[(aa,bb,mm,nn)]] )\
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eege_index[(aa-1,bb,mm,nn)]] \
                                                                                                  + self.cg[(aa,1)] * Sv[self.eege_index[(aa+1,bb,mm,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.ggge_index[(aa,bb+1,mm,nn)]] \
                                                                                                  + self.cg[(bb-1,1)] * Sv[self.ggge_index[(aa,bb-1,mm,nn)]] ) \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(mm,-1)] * Sv[self.geee_index[(aa,bb,mm-1,nn)]] \
                                                                                                  + self.cg[(mm,1)] * Sv[self.geee_index[(aa,bb,mm+1,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(nn+1,-1)] * Sv[self.gegg_index[(aa,bb,mm,nn+1)]] \
                                                                                                  + self.cg[(nn-1,1)] * Sv[self.gegg_index[(aa,bb,mm,nn-1)]] ) \
                                                                    )
                
                        # Cavity
                        dSv[self.gege_index[(aa,bb,mm,nn)]] += 1j * chitildeN/2.0 * (2 * self.cg[(aa,0)] * three_minus('ee','ge','Pminus',(aa,bb),(mm,nn)) \
                                                                                    - 2 * self.cg[(bb,0)] * three_minus('gg','ge','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + 2 * self.cg[(mm,0)] * three_minus('ge','ee','Pminus',(aa,bb),(mm,nn)) \
                                                                                    - 2 * self.cg[(nn,0)] * three_minus('ge','gg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,-1)] * three_minus('ee','ge','Sminus',(aa-1,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,1)] * three_minus('ee','ge','Sminus',(aa+1,bb),(mm,nn)) \
                                                                                    - self.cg[(bb+1,-1)] * three_minus('gg','ge','Sminus',(aa,bb+1),(mm,nn)) \
                                                                                    - self.cg[(bb-1,1)] * three_minus('gg','ge','Sminus',(aa,bb-1),(mm,nn)) \
                                                                                    + self.cg[(mm,-1)] * three_minus('ge','ee','Sminus',(aa,bb),(mm-1,nn)) \
                                                                                    + self.cg[(mm,1)] * three_minus('ge','ee','Sminus',(aa,bb),(mm+1,nn)) \
                                                                                    - self.cg[(nn+1,-1)] * three_minus('ge','gg','Sminus',(aa,bb),(mm,nn+1)) \
                                                                                    - self.cg[(nn-1,1)] * three_minus('ge','gg','Sminus',(aa,bb),(mm,nn-1)) \
                                                                                    )
        
        # geeg
        for aa in self.Mgs:
            for bb in self.Mes:
                for mm in self.Mes:
                    for nn in self.Mgs:
                        
                        # B-field and energies
                        dSv[self.geeg_index[(aa,bb,mm,nn)]] += 1j * ( param.zeeman_g * float(aa-nn) + param.zeeman_e * float(mm-bb) ) * Sv[self.geeg_index[(aa,bb,mm,nn)]]
                
                        # Spontaneous emission
                        temp = 0
                        for q in self.Qs:   temp += 2 * self.cg[(nn,q)] * self.cg[(aa,q)] * oneoverN * self.kron_del(bb,mm) * Sv[self.ee_index[(aa+q,nn+q)]] \
                                                    + 2 * self.cg[(mm-q,q)] * self.cg[(bb-q,q)] * oneoverN * self.kron_del(bb,mm) * Sv[self.gg_index[(aa,nn)]] \
                                                    - (self.cg[(bb-q,q)]**2 + self.cg[(mm-q,q)]**2) * Sv[self.geeg_index[(aa,bb,mm,nn)]]
                        dSv[self.geeg_index[(aa,bb,mm,nn)]] += param.gamma0/2 * temp
                
                        # Driving
                        dSv[self.geeg_index[(aa,bb,mm,nn)]] += 1j * (param.rabi_Pi * ( self.cg[(aa,0)] * Sv[self.eeeg_index[(aa,bb,mm,nn)]] \
                                                                                     - self.cg[(bb,0)] * Sv[self.ggeg_index[(aa,bb,mm,nn)]] )\
                                                                     + np.conj(param.rabi_Pi) * ( self.cg[(mm,0)] * Sv[self.gegg_index[(aa,bb,mm,nn)]] \
                                                                                                - self.cg[(nn,0)] * Sv[self.geee_index[(aa,bb,mm,nn)]] ) \
                                                                     + param.rabi_Sigma/sqrt(2) * ( self.cg[(aa,-1)] * Sv[self.eeeg_index[(aa-1,bb,mm,nn)]] \
                                                                                                  + self.cg[(aa,1)] * Sv[self.eeeg_index[(aa+1,bb,mm,nn)]] ) \
                                                                     - param.rabi_Sigma/sqrt(2) * ( self.cg[(bb+1,-1)] * Sv[self.ggeg_index[(aa,bb+1,mm,nn)]] \
                                                                                                  + self.cg[(bb-1,1)] * Sv[self.ggeg_index[(aa,bb-1,mm,nn)]] ) \
                                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(mm+1,-1)] * Sv[self.gegg_index[(aa,bb,mm+1,nn)]] \
                                                                                                           + self.cg[(mm-1,1)] * Sv[self.gegg_index[(aa,bb,mm-1,nn)]] ) \
                                                                     - np.conj(param.rabi_Sigma)/sqrt(2) * ( self.cg[(nn,-1)] * Sv[self.geee_index[(aa,bb,mm,nn-1)]] \
                                                                                                           + self.cg[(nn,1)] * Sv[self.geee_index[(aa,bb,mm,nn+1)]] ) \
                                                                    )
                
                        # Cavity
                        dSv[self.geeg_index[(aa,bb,mm,nn)]] += 1j * chitildeN/2.0 * (2 * self.cg[(aa,0)] * three_minus('ee','eg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    - 2 * self.cg[(bb,0)] * three_minus('gg','eg','Pminus',(aa,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,-1)] * three_minus('ee','eg','Sminus',(aa-1,bb),(mm,nn)) \
                                                                                    + self.cg[(aa,1)] * three_minus('ee','eg','Sminus',(aa+1,bb),(mm,nn)) \
                                                                                    - self.cg[(bb+1,-1)] * three_minus('gg','eg','Sminus',(aa,bb+1),(mm,nn)) \
                                                                                    - self.cg[(bb-1,1)] * three_minus('gg','eg','Sminus',(aa,bb-1),(mm,nn)) \
                                                                                    )\
                                                             + 1j * np.conj(chitildeN)/2.0 * ( 2 * self.cg[(mm,0)] * three_plus('Pplus','ge','gg',(aa,bb),(mm,nn)) \
                                                                                             - 2 * self.cg[(nn,0)] * three_plus('Pplus','ge','ee',(aa,bb),(mm,nn)) \
                                                                                             + self.cg[(mm+1,-1)] * three_plus('Splus','ge','gg',(aa,bb),(mm+1,nn)) \
                                                                                             + self.cg[(mm-1,1)] * three_plus('Splus','ge','gg',(aa,bb),(mm-1,nn)) \
                                                                                             - self.cg[(nn,-1)] * three_plus('Splus','ge','ee',(aa,bb),(mm,nn-1)) \
                                                                                             - self.cg[(nn,1)] * three_plus('Splus','ge','ee',(aa,bb),(mm,nn+1)) \
                                                                                             )
                                                                                             
                        """
                        # Symmetrize BBGKY
                        if param.method=='bbgky':
                            dSv[self.geeg_index[(aa,bb,mm,nn)]] += oneover2N * self.kron_del(nn,aa) * dSv[self.ee_index[(mm,bb)]] \
                                                                    - oneover2N * self.kron_del(bb,mm) * dSv[self.gg_index[(aa,nn)]]
                        #"""
        
        
        
        return dSv.reshape( (len(dSv)*len(dSv[0])) )
    
    
    
    
    def fill_auxiliary_variables (self,mainSv):
        """
        Return array with auxiliary variables computed from array Sv of variables.
        The auxliary variables include: redundant orderings of g and e, as well as correlators of Pi/Sigma
        """
        Sv = np.concatenate( ( mainSv , np.zeros((self.n_auxvariables,self.iterations),dtype=complex) ), axis=0)     # Each column is a classical trajectory in TWA
        
        ######
        ######      JUST FOR TESTING: COMPARISON WITH MEAN-FIELD // REMOVE FOR REAL COMPUTATIONS!!!!!!
        ######
        
        """
        oneminus1overN = 1-1/self.Ntotal
        oneoverN = 1/self.Ntotal
        
        
        # 2pt functions
        for aa in self.Mgs:
            for bb in self.Mgs:
                for mm in self.Mgs:
                    for nn in self.Mgs:
                        Sv[ self.gggg_index[(aa,bb,mm,nn)] ] = oneminus1overN * Sv[ self.gg_index[(aa,bb)] ] * Sv[ self.gg_index[(mm,nn)] ] \
                                                                    + oneoverN * self.kron_del(bb,mm) * Sv[ self.gg_index[(aa,nn)] ]
        for aa in self.Mes:
            for bb in self.Mes:
                for mm in self.Mes:
                    for nn in self.Mes:
                        Sv[ self.eeee_index[(aa,bb,mm,nn)] ] = oneminus1overN * Sv[ self.ee_index[(aa,bb)] ] * Sv[ self.ee_index[(mm,nn)] ] \
                                                                    + oneoverN * self.kron_del(bb,mm) * Sv[ self.ee_index[(aa,nn)] ]
        for aa in self.Mgs:
            for bb in self.Mgs:
                for mm in self.Mes:
                    for nn in self.Mes:
                        Sv[ self.ggee_index[(aa,bb,mm,nn)] ] = oneminus1overN * Sv[ self.gg_index[(aa,bb)] ] * Sv[ self.ee_index[(mm,nn)] ]
        for aa in self.Mgs:
            for bb in self.Mgs:
                for mm in self.Mgs:
                    for nn in self.Mes:
                        Sv[ self.ggge_index[(aa,bb,mm,nn)] ] = oneminus1overN * Sv[ self.gg_index[(aa,bb)] ] * Sv[ self.ge_index[(mm,nn)] ] \
                                                                    + oneoverN * self.kron_del(bb,mm) * Sv[ self.ge_index[(aa,nn)] ]
        for aa in self.Mgs:
            for bb in self.Mes:
                for mm in self.Mes:
                    for nn in self.Mes:
                        Sv[ self.geee_index[(aa,bb,mm,nn)] ] = oneminus1overN * Sv[ self.ge_index[(aa,bb)] ] * Sv[ self.ee_index[(mm,nn)] ] \
                                                                    + oneoverN * self.kron_del(bb,mm) * Sv[ self.ge_index[(aa,nn)] ]
        for aa in self.Mgs:
            for bb in self.Mes:
                for mm in self.Mgs:
                    for nn in self.Mes:
                        Sv[ self.gege_index[(aa,bb,mm,nn)] ] = oneminus1overN * Sv[ self.ge_index[(aa,bb)] ] * Sv[ self.ge_index[(mm,nn)] ]
        for aa in self.Mgs:
            for bb in self.Mes:
                for mm in self.Mes:
                    for nn in self.Mgs:
                        Sv[ self.geeg_index[(aa,bb,mm,nn)] ] = oneminus1overN * Sv[ self.ge_index[(aa,bb)] ] * np.conj(Sv[ self.ge_index[(nn,mm)] ]) \
                                                                    + oneoverN * self.kron_del(bb,mm) * Sv[ self.gg_index[(aa,nn)] ]
        """
        
        
        ######
        ######      VARIABLES with "ge" in DIFFERENT ORDERINGS
        ######
        
        
        if param.method=='cumulant':
            
            oneminus1overN = 1-1/self.Ntotal
            oneoverN = 1/self.Ntotal
            
        if param.method=='bbgky':
            
            #oneminus1overN = 1
            #oneoverN = 0
            
            oneminus1overN = 1-1/self.Ntotal
            oneoverN = 1/self.Ntotal
            
        # 1pt extra
        for aa in self.Mes:
            for bb in self.Mgs:
                Sv[ self.eg_index[(aa,bb)] ] = np.conj( Sv[ self.ge_index[(bb,aa)] ] )
     
        # 2pt extra
        for aa in self.Mes:
            for bb in self.Mes:
                for mm in self.Mgs:
                    for nn in self.Mgs:
                        Sv[ self.eegg_index[(aa,bb,mm,nn)] ] = Sv[ self.ggee_index[(mm,nn,aa,bb)] ]
        for aa in self.Mgs:
            for bb in self.Mgs:
                for mm in self.Mes:
                    for nn in self.Mgs:
                        Sv[ self.ggeg_index[(aa,bb,mm,nn)] ] = np.conj( Sv[ self.ggge_index[(bb,aa,nn,mm)] ] \
                                                                    - oneoverN * self.kron_del(aa,nn) * Sv[ self.ge_index[(bb,mm)] ] )
        for aa in self.Mgs:
            for bb in self.Mes:
                for mm in self.Mgs:
                    for nn in self.Mgs:
                        Sv[ self.gegg_index[(aa,bb,mm,nn)] ] = Sv[ self.ggge_index[(mm,nn,aa,bb)] ] \
                                                                    - oneoverN * self.kron_del(nn,aa) * Sv[ self.ge_index[(mm,bb)] ]
        for aa in self.Mes:
            for bb in self.Mgs:
                for mm in self.Mgs:
                    for nn in self.Mgs:
                        Sv[ self.eggg_index[(aa,bb,mm,nn)] ] = np.conj( Sv[ self.ggge_index[(nn,mm,bb,aa)] ] )
        for aa in self.Mes:
            for bb in self.Mgs:
                for mm in self.Mes:
                    for nn in self.Mes:
                        Sv[ self.egee_index[(aa,bb,mm,nn)] ] = np.conj( Sv[ self.geee_index[(bb,aa,nn,mm)] ] \
                                                                            - oneoverN * self.kron_del(aa,nn) * Sv[ self.ge_index[(bb,mm)] ] )
        for aa in self.Mes:
            for bb in self.Mes:
                for mm in self.Mgs:
                    for nn in self.Mes:
                        Sv[ self.eege_index[(aa,bb,mm,nn)] ] = Sv[ self.geee_index[(mm,nn,aa,bb)] ] \
                                                                    - oneoverN * self.kron_del(aa,nn) * Sv[ self.ge_index[(mm,bb)] ]
        for aa in self.Mes:
            for bb in self.Mes:
                for mm in self.Mes:
                    for nn in self.Mgs:
                        Sv[ self.eeeg_index[(aa,bb,mm,nn)] ] = np.conj( Sv[ self.geee_index[(nn,mm,bb,aa)] ] )
        for aa in self.Mes:
            for bb in self.Mgs:
                for mm in self.Mes:
                    for nn in self.Mgs:
                        Sv[ self.egeg_index[(aa,bb,mm,nn)] ] = np.conj( Sv[ self.gege_index[(nn,mm,bb,aa)] ] )
        for aa in self.Mes:
            for bb in self.Mgs:
                for mm in self.Mgs:
                    for nn in self.Mes:
                        Sv[ self.egge_index[(aa,bb,mm,nn)] ] = Sv[ self.geeg_index[(mm,nn,aa,bb)] ] \
                                                                    + oneoverN * self.kron_del(bb,mm) * Sv[ self.ee_index[(aa,nn)] ] \
                                                                    - oneoverN * self.kron_del(aa,nn) * Sv[ self.gg_index[(mm,bb)] ]
        
        
        
        ######
        ######      VARIABLES with PI and SIGMA
        ######
        
        # 1pt Pi and Sigma (Note normalization of Sigma)
        Sv[self.Pminus_index] = 0
        Sv[self.Sminus_index] = 0
        for mm in self.Mgs:
            Sv[self.Pminus_index] += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
            Sv[self.Sminus_index] += self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]] + self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
        Sv[self.Pplus_index] = np.conj(Sv[self.Pminus_index])
        Sv[self.Splus_index] = np.conj(Sv[self.Sminus_index])
        
        # 2pt Pi and Sigma (THIS CAN BE WRITTEN MORE COMPACTLY USING self.index and running over g,e)
        #"""
        for la in ['g','e']:
            for lb in ['g','e']:
                for aa in self.Ms[la]:
                    for bb in self.Ms[lb]:
                        Sv[self.index[la+lb+'Pminus'][(aa,bb)]] = 0       # self.index['ggPminus'][(aa,bb)]
                        Sv[self.index[la+lb+'Sminus'][(aa,bb)]] = 0
                        for mm in self.Mgs:
                            Sv[self.index[la+lb+'Pminus'][(aa,bb)]] += self.cg[(mm,0)] * Sv[self.index[la+lb+'ge'][(aa,bb,mm,mm)]]
                            Sv[self.index[la+lb+'Sminus'][(aa,bb)]] += self.cg[(mm,-1)] * Sv[self.index[la+lb+'ge'][(aa,bb,mm,mm-1)]] \
                                                                        + self.cg[(mm,1)] * Sv[self.index[la+lb+'ge'][(aa,bb,mm,mm+1)]]
        for la in ['g','e']:
            for lb in ['g','e']:
                for aa in self.Ms[la]:
                    for bb in self.Ms[lb]:
                        Sv[self.index['Pplus'+la+lb][(aa,bb)]] = np.conj(Sv[self.index[lb+la+'Pminus'][(bb,aa)]])
                        Sv[self.index['Splus'+la+lb][(aa,bb)]] = np.conj(Sv[self.index[lb+la+'Sminus'][(bb,aa)]])
        #"""
        """
        for aa in self.Mgs:
            for bb in self.Mgs:
                Sv[self.gg_Pminus_index[(aa,bb)]] = 0
                Sv[self.gg_Sminus_index[(aa,bb)]] = 0
                for mm in self.Mgs:
                    Sv[self.gg_Pminus_index[(aa,bb)]] += self.cg[(mm,0)] * Sv[self.ggge_index[(aa,bb,mm,mm)]]
                    Sv[self.gg_Sminus_index[(aa,bb)]] += self.cg[(mm,-1)] * Sv[self.ggge_index[(aa,bb,mm,mm-1)]] + self.cg[(mm,1)] * Sv[self.ggge_index[(aa,bb,mm,mm+1)]]
        for aa in self.Mes:
            for bb in self.Mes:
                Sv[self.ee_Pminus_index[(aa,bb)]] = 0
                Sv[self.ee_Sminus_index[(aa,bb)]] = 0
                for mm in self.Mgs:
                    Sv[self.ee_Pminus_index[(aa,bb)]] += self.cg[(mm,0)] * Sv[self.eege_index[(aa,bb,mm,mm)]]
                    Sv[self.ee_Sminus_index[(aa,bb)]] += self.cg[(mm,-1)] * Sv[self.eege_index[(aa,bb,mm,mm-1)]] + self.cg[(mm,1)] * Sv[self.eege_index[(aa,bb,mm,mm+1)]]
        for aa in self.Mgs:
            for bb in self.Mes:
                Sv[self.ge_Pminus_index[(aa,bb)]] = 0
                Sv[self.ge_Sminus_index[(aa,bb)]] = 0
                for mm in self.Mgs:
                    Sv[self.ge_Pminus_index[(aa,bb)]] += self.cg[(mm,0)] * Sv[self.gege_index[(aa,bb,mm,mm)]]
                    Sv[self.ge_Sminus_index[(aa,bb)]] += self.cg[(mm,-1)] * Sv[self.gege_index[(aa,bb,mm,mm-1)]] + self.cg[(mm,1)] * Sv[self.gege_index[(aa,bb,mm,mm+1)]]
        for aa in self.Mes:
            for bb in self.Mgs:
                Sv[self.eg_Pminus_index[(aa,bb)]] = 0
                Sv[self.eg_Sminus_index[(aa,bb)]] = 0
                for mm in self.Mgs:
                    Sv[self.eg_Pminus_index[(aa,bb)]] += self.cg[(mm,0)] * Sv[self.egge_index[(aa,bb,mm,mm)]]
                    Sv[self.eg_Sminus_index[(aa,bb)]] += self.cg[(mm,-1)] * Sv[self.egge_index[(aa,bb,mm,mm-1)]] + self.cg[(mm,1)] * Sv[self.egge_index[(aa,bb,mm,mm+1)]]
                    
        for aa in self.Mgs:
            for bb in self.Mgs:
                Sv[self.Pplus_gg_index[(aa,bb)]] = np.conj(Sv[self.gg_Pminus_index[(bb,aa)]])
                Sv[self.Splus_gg_index[(aa,bb)]] = np.conj(Sv[self.gg_Sminus_index[(bb,aa)]])
        for aa in self.Mes:
            for bb in self.Mes:
                Sv[self.Pplus_ee_index[(aa,bb)]] = np.conj(Sv[self.ee_Pminus_index[(bb,aa)]])
                Sv[self.Splus_ee_index[(aa,bb)]] = np.conj(Sv[self.ee_Sminus_index[(bb,aa)]])
        for aa in self.Mgs:
            for bb in self.Mes:
                Sv[self.Pplus_ge_index[(aa,bb)]] = np.conj(Sv[self.eg_Pminus_index[(bb,aa)]])
                Sv[self.Splus_ge_index[(aa,bb)]] = np.conj(Sv[self.eg_Sminus_index[(bb,aa)]])
        for aa in self.Mes:
            for bb in self.Mgs:
                Sv[self.Pplus_eg_index[(aa,bb)]] = np.conj(Sv[self.ge_Pminus_index[(bb,aa)]])
                Sv[self.Splus_eg_index[(aa,bb)]] = np.conj(Sv[self.ge_Sminus_index[(bb,aa)]])
        #"""   
        
        
        
        # Return array containing main and auxiliary variables
        return Sv
        
    
    

    '''
    ### Third version (without symmetrization corrections)
    def MF_eqs (self,t,Svtemp):
        """Computes and returns right-hand side of mean-field equations"""
        Sv = Svtemp.reshape( (self.nvariables,self.iterations) )
        dSv = np.zeros((self.nvariables,self.iterations),dtype=complex)
        
        chitildeN = self.Ntotal * (param.g_coh - 1j*param.g_inc/2)
        
        ############
        ####            IDEA: Avoid all the if clauses by defining all entries of Sv and CG that are out of bounds as 0
        ############
        
        Pi_minus = 0
        Sigma_minus = 0
        for mm in self.Mgs: Pi_minus += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
            #if mm in self.Mes: Pi_minus += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
        for mm in self.Mgs: Sigma_minus += 0.5 * self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]] + 0.5 * self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
            #if mm-1 in self.Mes: Sigma_minus += 0.5 * self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]]
            #if mm+1 in self.Mes: Sigma_minus += 0.5 * self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
        Pi_plus = np.conj(Pi_minus)
        Sigma_plus = np.conj(Sigma_minus)
        
        
        # gg
        
        #for ii in range(100):
        #    aabb = [[aa,bb] for aa in self.Mgs for bb in self.Mgs]
        #    ggs = [ self.gg_index[(aa,bb)] for aa,bb in aabb ]
        #    dSv[ggs] += 1j * param.zeeman_g * np.array([float(aa-bb) for aa,bb in aabb]) * Sv[ggs]
        
        #for ii in range(100):
        #    for aa in self.Mgs:
        #        for bb in self.Mgs:
        #            dSv[self.gg_index[(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.gg_index[(aa,bb)]]
        
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.gg_index[(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.gg_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(aa,q)] * self.cg[(bb,q)] * Sv[self.ee_index[(aa+q,bb+q)]]
                    #if aa+q in self.Mes:
                    #    if bb+q in self.Mes:
                    #        temp += self.cg[(aa,q)] * self.cg[(bb,q)] * Sv[self.ee_index[(aa+q,bb+q)]]
                dSv[self.gg_index[(aa,bb)]] += param.gamma0 * temp
                
                # Driving
                dSv[self.gg_index[(aa,bb)]] += 1j * ( param.rabi_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                      - np.conj(param.rabi_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                      + param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])\
                                                      + param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])\
                                                      - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]\
                                                      - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]\
                                                    )
                
                #temp = 0
                #temp += param.rabi_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                #temp += - np.conj(param.rabi_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]
                #temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])
                #temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])
                #temp += - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]
                #temp += - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]
                
                #dSv[self.gg_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                dSv[self.gg_index[(aa,bb)]] += 1j * chitildeN * ( Pi_minus * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                                  + Sigma_minus * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])\
                                                                  + Sigma_minus * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])\
                                                                )
                
                #temp = 0
                #temp += Pi_minus * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                #temp += Sigma_minus * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])
                #temp += Sigma_minus * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])
                
                #dSv[self.gg_index[(aa,bb)]] += 1j * chitildeN * temp
                
                
                dSv[self.gg_index[(aa,bb)]] += -1j * np.conj(chitildeN) * (Pi_plus * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                                           + Sigma_plus * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]\
                                                                           + Sigma_plus * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]\
                                                                          )
                
                #temp = 0
                #temp += Pi_plus * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]
                #temp += Sigma_plus * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]
                #temp += Sigma_plus * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]
                        
                #dSv[self.gg_index[(aa,bb)]] += -1j * np.conj(chitildeN) * temp
                
        
                
        # ee
        for aa in self.Mes:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ee_index[(aa,bb)]] += 1j * param.zeeman_e * float(aa-bb) * Sv[self.ee_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(aa-q,q)]**2 * Sv[self.ee_index[(aa,bb)]] + self.cg[(bb-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                    #if aa-q in self.Mgs: temp += self.cg[(aa-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                    #if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                            
                dSv[self.ee_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                dSv[self.ee_index[(aa,bb)]] += 1j * (- param.rabi_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                     + np.conj(param.rabi_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])\
                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]\
                                                     + np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]\
                                                    )
                
                #temp = 0
                #temp += - param.rabi_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                #temp += np.conj(param.rabi_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]
                #temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])
                #temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])
                #temp += np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]
                #temp += np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]
                
                #dSv[self.ee_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                dSv[self.ee_index[(aa,bb)]] += - 1j * chitildeN * (Pi_minus * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])\
                                                                   + Sigma_minus * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])\
                                                                   + Sigma_minus * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])\
                                                                  )
                
                #temp = 0
                #temp += Pi_minus * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                #temp += Sigma_minus * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])
                #temp += Sigma_minus * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])
                    
                #dSv[self.ee_index[(aa,bb)]] += - 1j * chitildeN * temp
                
                
                dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitildeN) * (Pi_plus * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]\
                                                                          + Sigma_plus * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]\
                                                                          + Sigma_plus * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]\
                                                                         )
                
                #temp = 0
                #temp += Pi_plus * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]
                #temp += Sigma_plus * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]
                #temp += Sigma_plus * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]
                        
                #dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitildeN) * temp
                
                
        
        # ge
        for aa in self.Mgs:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.zeeman_g*float(aa) - param.zeeman_e*float(bb) + param.detuning) * Sv[self.ge_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:   temp += self.cg[(bb-q,q)]**2 * Sv[self.ge_index[(aa,bb)]]
                    #if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ge_index[(aa,bb)]]
                            
                dSv[self.ge_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.rabi_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]\
                                                     - param.rabi_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]\
                                                     + param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]\
                                                     + param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]\
                                                     - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]\
                                                    )
                
                #temp = 0
                #temp += param.rabi_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]
                #temp += - param.rabi_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]
                #temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]
                #temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]
                #temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]
                #temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]

                #dSv[self.ge_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                dSv[self.ge_index[(aa,bb)]] += 1j * chitildeN * (Pi_minus * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]\
                                                                 - Pi_minus * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]\
                                                                 + Sigma_minus * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]\
                                                                 + Sigma_minus * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]\
                                                                 - Sigma_minus * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]\
                                                                 - Sigma_minus * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]
                                                                )
                
                #temp = 0
                #temp += Pi_minus * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]
                #temp += - Pi_minus * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]
                
                #temp += Sigma_minus * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]
                #temp += Sigma_minus * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]
                #temp += - Sigma_minus * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]
                #temp += - Sigma_minus * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]
                    
                #dSv[self.ge_index[(aa,bb)]] += 1j * chitildeN * temp
        
        
        
        return dSv.reshape( (len(dSv)*len(dSv[0])) )
    '''
    
    
    '''
    ### Second version
    def MF_eqs (self,t,Sv):
        """Computes and returns right-hand side of mean-field equations"""
        dSv = np.zeros((self.nvariables),dtype=complex)        # Later the "1" will become "iterations" for TWA
        chitildeN = self.Ntotal * (param.g_coh - 1j*param.g_inc/2)
        
        ############
        ####            IDEA: Avoid all the if clauses by defining all entries of Sv and CG that are out of bounds as 0
        ############
        
        Pi_minus = 0
        Sigma_minus = 0
        for mm in self.Mgs:
            if mm in self.Mes: Pi_minus += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
        for mm in self.Mgs:
            if mm-1 in self.Mes: Sigma_minus += 0.5 * self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]]
            if mm+1 in self.Mes: Sigma_minus += 0.5 * self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
        Pi_plus = np.conj(Pi_minus)
        Sigma_plus = np.conj(Sigma_minus)
        
        
        # gg
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.gg_index[(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.gg_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:
                    if aa+q in self.Mes:
                        if bb+q in self.Mes:
                            temp += self.cg[(aa,q)] * self.cg[(bb,q)] * Sv[self.ee_index[(aa+q,bb+q)]]
                dSv[self.gg_index[(aa,bb)]] += param.gamma0 * temp
                
                # Driving
                temp = 0
                if aa in self.Mes: temp += param.rabi_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                if bb in self.Mes: temp += - np.conj(param.rabi_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]
                if aa-1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])
                if aa+1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])
                if bb-1 in self.Mes: temp += - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]
                if bb+1 in self.Mes: temp += - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]
                dSv[self.gg_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                temp = 0
                if aa in self.Mes:  temp += Pi_minus * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                if (aa-1 in self.Mes):  temp += Sigma_minus * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])
                if (aa+1 in self.Mes):  temp += Sigma_minus * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])
                
                dSv[self.gg_index[(aa,bb)]] += 1j * chitildeN * temp
                
                temp = 0
                if bb in self.Mes:  temp += Pi_plus * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]
                if (bb-1 in self.Mes):  temp += Sigma_plus * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]
                if (bb+1 in self.Mes):  temp += Sigma_plus * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]
                        
                dSv[self.gg_index[(aa,bb)]] += -1j * np.conj(chitildeN) * temp
                
        
                
        # ee
        for aa in self.Mes:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ee_index[(aa,bb)]] += 1j * param.zeeman_e * float(aa-bb) * Sv[self.ee_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:
                    if aa-q in self.Mgs: temp += self.cg[(aa-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                    if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                            
                dSv[self.ee_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                temp = 0
                if bb in self.Mgs: temp += - param.rabi_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                if aa in self.Mgs: temp += np.conj(param.rabi_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]
                if bb+1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])
                if bb-1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])
                if aa+1 in self.Mgs: temp += np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]
                if aa-1 in self.Mgs: temp += np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]
                dSv[self.ee_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                temp = 0
                if bb in self.Mgs:  temp += Pi_minus * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                if (bb+1 in self.Mgs):  temp += Sigma_minus * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])
                if (bb-1 in self.Mgs):  temp += Sigma_minus * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])
                    
                dSv[self.ee_index[(aa,bb)]] += - 1j * chitildeN * temp
                
                temp = 0
                if aa in self.Mgs:  temp += Pi_plus * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]
                if (aa+1 in self.Mgs):  temp += Sigma_plus * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]
                if (aa-1 in self.Mgs):  temp += Sigma_plus * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]
                        
                dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitildeN) * temp
                
                
        
        # ge
        for aa in self.Mgs:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.zeeman_g*float(aa) - param.zeeman_e*float(bb) + param.detuning) * Sv[self.ge_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:
                    if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ge_index[(aa,bb)]]
                            
                dSv[self.ge_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                temp = 0
                if aa in self.Mes: temp += param.rabi_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]
                if bb in self.Mgs: temp += - param.rabi_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]
                if aa-1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]
                if aa+1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]
                if bb+1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]
                if bb-1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]
                dSv[self.ge_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                temp = 0
                if aa in self.Mes: temp += Pi_minus * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]
                if bb in self.Mgs: temp += - Pi_minus * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]
                
                if aa-1 in self.Mes: temp += Sigma_minus * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]
                if aa+1 in self.Mes: temp += Sigma_minus * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]
                if bb+1 in self.Mgs: temp += - Sigma_minus * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]
                if bb-1 in self.Mgs: temp += - Sigma_minus * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]
                    
                dSv[self.ge_index[(aa,bb)]] += 1j * chitildeN * temp
        
        
        
        return dSv
    '''

    '''
       
    ### First version 
    def MF_eqs (self,t,Sv):
        """Computes and returns right-hand side of mean-field equations"""
        dSv = np.zeros((self.nvariables),dtype=complex)        # Later the "1" will become "iterations" for TWA
        chitildeN = self.Ntotal * (param.g_coh - 1j*param.g_inc/2)
        
        ############
        ####            IDEA: Avoid all the if clauses by defining all entries of Sv and CG that are out of bounds as 0
        ############
        
        
        # gg
        for aa in self.Mgs:
            for bb in self.Mgs:
                
                # B-field and energies
                dSv[self.gg_index[(aa,bb)]] += 1j * param.zeeman_g * float(aa-bb) * Sv[self.gg_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:
                    if aa+q in self.Mes:
                        if bb+q in self.Mes:
                            temp += self.cg[(aa,q)] * self.cg[(bb,q)] * Sv[self.ee_index[(aa+q,bb+q)]]
                dSv[self.gg_index[(aa,bb)]] += param.gamma0 * temp
                
                # Driving
                temp = 0
                if aa in self.Mes: temp += param.rabi_Pi * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                if bb in self.Mes: temp += - np.conj(param.rabi_Pi) * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]
                if aa-1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])
                if aa+1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])
                if bb-1 in self.Mes: temp += - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]
                if bb+1 in self.Mes: temp += - np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]
                dSv[self.gg_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                temp = 0
                sumtemp = 0
                if aa in self.Mes:
                    for mm in self.Mgs:
                        if mm in self.Mes: sumtemp += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
                    temp += sumtemp * self.cg[(aa,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                sumtemp = 0
                if (aa-1 in self.Mes) or (aa+1 in self.Mes):
                    for mm in self.Mgs:
                        if mm-1 in self.Mes: sumtemp += self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]]
                        if mm+1 in self.Mes: sumtemp += self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
                if (aa-1 in self.Mes):
                    temp += 0.5 * sumtemp * self.cg[(aa,-1)] * np.conj(Sv[self.ge_index[(bb,aa-1)]])
                if (aa+1 in self.Mes):
                    temp += 0.5 * sumtemp * self.cg[(aa,1)] * np.conj(Sv[self.ge_index[(bb,aa+1)]])
                    
                dSv[self.gg_index[(aa,bb)]] += 1j * chitildeN * temp
                
                temp = 0
                sumtemp = 0
                if bb in self.Mes:
                    for mm in self.Mgs:
                        if mm in self.Mes: sumtemp += self.cg[(mm,0)] * np.conj(Sv[self.ge_index[(mm,mm)]])
                    temp += sumtemp * self.cg[(bb,0)] * Sv[self.ge_index[(aa,bb)]]
                sumtemp = 0
                if (bb-1 in self.Mes) or (bb+1 in self.Mes):
                    for mm in self.Mgs:
                        if mm-1 in self.Mes: sumtemp += self.cg[(mm,-1)] * np.conj(Sv[self.ge_index[(mm,mm-1)]])
                        if mm+1 in self.Mes: sumtemp += self.cg[(mm,1)] * np.conj(Sv[self.ge_index[(mm,mm+1)]])
                if (bb-1 in self.Mes):
                    temp += 0.5 * sumtemp * self.cg[(bb,-1)] * Sv[self.ge_index[(aa,bb-1)]]
                if (bb+1 in self.Mes):
                    temp += 0.5 * sumtemp * self.cg[(bb,1)] * Sv[self.ge_index[(aa,bb+1)]]
                        
                dSv[self.gg_index[(aa,bb)]] += -1j * np.conj(chitildeN) * temp
                
        
                
        # ee
        for aa in self.Mes:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ee_index[(aa,bb)]] += 1j * param.zeeman_e * float(aa-bb) * Sv[self.ee_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:
                    if aa-q in self.Mgs: temp += self.cg[(aa-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                    if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ee_index[(aa,bb)]]
                            
                dSv[self.ee_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                temp = 0
                if bb in self.Mgs: temp += - param.rabi_Pi * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                if aa in self.Mgs: temp += np.conj(param.rabi_Pi) * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]
                if bb+1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])
                if bb-1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])
                if aa+1 in self.Mgs: temp += np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]
                if aa-1 in self.Mgs: temp += np.conj(param.rabi_Sigma)/sqrt(2) * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]
                dSv[self.ee_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                temp = 0
                sumtemp = 0
                if bb in self.Mgs:
                    for mm in self.Mgs:
                        if mm in self.Mes: sumtemp += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
                    temp += sumtemp * self.cg[(bb,0)] * np.conj(Sv[self.ge_index[(bb,aa)]])
                sumtemp = 0
                if (bb-1 in self.Mgs) or (bb+1 in self.Mgs):
                    for mm in self.Mgs:
                        if mm-1 in self.Mes: sumtemp += self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]]
                        if mm+1 in self.Mes: sumtemp += self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
                if (bb+1 in self.Mgs):
                    temp += 0.5 * sumtemp * self.cg[(bb+1,-1)] * np.conj(Sv[self.ge_index[(bb+1,aa)]])
                if (bb-1 in self.Mgs):
                    temp += 0.5 * sumtemp * self.cg[(bb-1,1)] * np.conj(Sv[self.ge_index[(bb-1,aa)]])
                    
                dSv[self.ee_index[(aa,bb)]] += - 1j * chitildeN * temp
                
                temp = 0
                sumtemp = 0
                if aa in self.Mgs:
                    for mm in self.Mgs:
                        if mm in self.Mes: sumtemp += self.cg[(mm,0)] * np.conj(Sv[self.ge_index[(mm,mm)]])
                    temp += sumtemp * self.cg[(aa,0)] * Sv[self.ge_index[(aa,bb)]]
                sumtemp = 0
                if (aa+1 in self.Mgs) or (aa-1 in self.Mgs):
                    for mm in self.Mgs:
                        if mm-1 in self.Mes: sumtemp += self.cg[(mm,-1)] * np.conj(Sv[self.ge_index[(mm,mm-1)]])
                        if mm+1 in self.Mes: sumtemp += self.cg[(mm,1)] * np.conj(Sv[self.ge_index[(mm,mm+1)]])
                if (aa+1 in self.Mgs):
                    temp += 0.5 * sumtemp * self.cg[(aa+1,-1)] * Sv[self.ge_index[(aa+1,bb)]]
                if (aa-1 in self.Mgs):
                    temp += 0.5 * sumtemp * self.cg[(aa-1,1)] * Sv[self.ge_index[(aa-1,bb)]]
                        
                dSv[self.ee_index[(aa,bb)]] += 1j * np.conj(chitildeN) * temp
                
                
        
        # ge
        for aa in self.Mgs:
            for bb in self.Mes:
                
                # B-field and energies
                dSv[self.ge_index[(aa,bb)]] += 1j * (param.zeeman_g*float(aa) - param.zeeman_e*float(bb) + param.detuning) * Sv[self.ge_index[(aa,bb)]]
                
                # Spontaneous emission
                temp = 0
                for q in self.Qs:
                    if bb-q in self.Mgs: temp += self.cg[(bb-q,q)]**2 * Sv[self.ge_index[(aa,bb)]]
                            
                dSv[self.ge_index[(aa,bb)]] += - param.gamma0/2 * temp
                
                # Driving
                temp = 0
                if aa in self.Mes: temp += param.rabi_Pi * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]
                if bb in self.Mgs: temp += - param.rabi_Pi * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]
                if aa-1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]
                if aa+1 in self.Mes: temp += param.rabi_Sigma/sqrt(2) * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]
                if bb+1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]
                if bb-1 in self.Mgs: temp += - param.rabi_Sigma/sqrt(2) * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]
                dSv[self.ge_index[(aa,bb)]] += 1j * temp
                
                # Cavity
                temp = 0
                sumtemp = 0
                for mm in self.Mgs:
                    if mm in self.Mes: sumtemp += self.cg[(mm,0)] * Sv[self.ge_index[(mm,mm)]]
                if aa in self.Mes: temp += sumtemp * self.cg[(aa,0)] * Sv[self.ee_index[(aa,bb)]]
                if bb in self.Mgs: temp += - sumtemp * self.cg[(bb,0)] * Sv[self.gg_index[(aa,bb)]]
                
                sumtemp = 0
                for mm in self.Mgs:
                    if mm-1 in self.Mes: sumtemp += self.cg[(mm,-1)] * Sv[self.ge_index[(mm,mm-1)]]
                    if mm+1 in self.Mes: sumtemp += self.cg[(mm,1)] * Sv[self.ge_index[(mm,mm+1)]]
                if aa-1 in self.Mes: temp += 0.5 * sumtemp * self.cg[(aa,-1)] * Sv[self.ee_index[(aa-1,bb)]]
                if aa+1 in self.Mes: temp += 0.5 * sumtemp * self.cg[(aa,1)] * Sv[self.ee_index[(aa+1,bb)]]
                if bb+1 in self.Mgs: temp += - 0.5 * sumtemp * self.cg[(bb+1,-1)] * Sv[self.gg_index[(aa,bb+1)]]
                if bb-1 in self.Mgs: temp += - 0.5 * sumtemp * self.cg[(bb-1,1)] * Sv[self.gg_index[(aa,bb-1)]]
                    
                dSv[self.ge_index[(aa,bb)]] += 1j * chitildeN * temp
        
        
        
        return dSv

    def truncate(self,number, digits) -> float:
        stepper = pow(10.0, digits)
        return math.trunc(stepper * number) / stepper
        
        
    '''
        
        
####################################################################

############                DYNAMICS                ###############

####################################################################
       
    def evolve_onestep (self):
        """ Evolve classical variables from t to t+dt. """
        self.solver.integrate(self.solver.t+param.dt)
        self.Sv = self.solver.y.reshape((self.nvariables,self.iterations))
        
        #print(self.Sv[self.ee_index[(0,0)]] + self.Sv[self.gg_index[(0,0)]])


    def set_solver (self,time):
        """ Choose solver for ODE and set initial conditions. """
        if param.method in ['MF','TWA']:
            self.solver = complex_ode(self.MF_eqs).set_integrator('dopri5',atol=param.atol,rtol=param.rtol)
        if param.method in ['cumulant','bbgky']:
            self.solver = complex_ode(self.cumulant_eqs).set_integrator('dopri5',atol=param.atol,rtol=param.rtol)
            
        print('atol:', self.solver._integrator.atol)
        print('rtol:', self.solver._integrator.rtol)
        
        self.solver.set_initial_value(self.Sv.reshape( (len(self.Sv)*len(self.Sv[0])) ), time)
        
        

        



####################################################################

#############                OUTPUT                ################

####################################################################
        
    
    def read_occs (self):
        """
        Outputs occupation of each level, summed over all atoms.
        """
        out = []
        
        # Occupations
        if 'populations' in param.which_observables:
            for aa in self.Mgs:
                out.append( np.sum(self.Sv[self.gg_index[(aa,aa)]].real)/self.iterations )
            for bb in self.Mes:
                out.append( np.sum(self.Sv[self.ee_index[(bb,bb)]].real)/self.iterations )
        
        if 'pop_variances' in param.which_observables:
            
            if param.method in ['MF','TWA']:
                
                # n^2 terms
                for aa in self.Mgs:
                    out.append( np.sum( (self.Sv[self.gg_index[(aa,aa)]].real)**2 )/self.iterations )
                for bb in self.Mes:
                    out.append( np.sum( (self.Sv[self.ee_index[(bb,bb)]].real)**2 )/self.iterations )
                    
                # Add occupation_aa * occupation_bb
                for aa in range(self.localhilbertsize):
                    for bb in range(aa+1,self.localhilbertsize):
                        
                        la = self.level_to_eg[aa]
                        lb = self.level_to_eg[bb]
                        ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                        mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                        
                        out.append( np.sum( self.Sv[self.index[la+la][(ma,ma)]].real * self.Sv[self.index[lb+lb][(mb,mb)]].real )/self.iterations )
                
                
            if param.method in ['cumulant','bbgky']:
                
                Sv = self.fill_auxiliary_variables(self.Sv)
                
                # n^2 terms
                for aa in self.Mgs:
                    out.append( np.sum( Sv[self.gggg_index[(aa,aa,aa,aa)]].real )/self.iterations )
                for bb in self.Mes:
                    out.append( np.sum( Sv[self.eeee_index[(bb,bb,bb,bb)]].real )/self.iterations )
                    
                # Add 1/2 ( occupation_aa * occupation_bb + occupation_bb * occupation_aa )
                for aa in range(self.localhilbertsize):
                    for bb in range(aa+1,self.localhilbertsize):
                        
                        la = self.level_to_eg[aa]
                        lb = self.level_to_eg[bb]
                        ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                        mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                        
                        out.append( np.sum( Sv[self.index[la+la+lb+lb][(ma,ma,mb,mb)]].real )/self.iterations )
                        
            
                
        if 'xyz' in param.which_observables:        #### CHANGE ORDERING TO FIT WITH ED!!!!!!
        
            for aa in range(self.localhilbertsize):
                for bb in range(aa+1,self.localhilbertsize):
                    
                    la = self.level_to_eg[aa]
                    lb = self.level_to_eg[bb]
                    ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                    mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                    
                    out.append( np.sum( 2*self.Sv[self.index[la+lb][(ma,mb)]].real ) / self.iterations )
                    out.append( np.sum( -2*self.Sv[self.index[la+lb][(ma,mb)]].imag ) / self.iterations )
                    out.append( np.sum( (self.Sv[self.index[lb+lb][(mb,mb)]]-self.Sv[self.index[la+la][(ma,ma)]]).real ) / self.iterations )
        
            """for aa in self.Mgs:
                for bb in self.Mgs:
                    if bb>aa:
                        out.append( np.sum( 2*self.Sv[self.gg_index[(aa,bb)]].real ) / self.iterations )
                        out.append( np.sum( -2*self.Sv[self.gg_index[(aa,bb)]].imag ) / self.iterations )
                        out.append( np.sum( (self.Sv[self.gg_index[(bb,bb)]]-self.Sv[self.gg_index[(aa,aa)]]).real ) / self.iterations )
                    
            for aa in self.Mgs:
                for bb in self.Mes:
                    out.append( np.sum( 2*self.Sv[self.ge_index[(aa,bb)]].real ) / self.iterations )
                    out.append( np.sum( -2*self.Sv[self.ge_index[(aa,bb)]].imag ) / self.iterations )
                    out.append( np.sum( (self.Sv[self.ee_index[(bb,bb)]]-self.Sv[self.gg_index[(aa,aa)]]).real ) / self.iterations )
                    
            for aa in self.Mes:
                for bb in self.Mes:
                    if bb>aa:
                        out.append( np.sum( 2*self.Sv[self.ee_index[(aa,bb)]].real ) / self.iterations )
                        out.append( np.sum( -2*self.Sv[self.ee_index[(aa,bb)]].imag ) / self.iterations )
                        out.append( np.sum( (self.Sv[self.ee_index[(bb,bb)]]-self.Sv[self.ee_index[(aa,aa)]]).real ) / self.iterations )"""
            
        
        if 'xyz_variances' in param.which_observables:
            print('ERROR: xyz_variances not implemented for MF/TWA/cumulant/BBGKY.')
        
        
        # Pi Sigma observables
        if 'photons' in param.which_observables:
            
            #######################
            ###     One-point   ###
            #######################
            # Pi and Sigma
            Pi_minus = 0
            Sigma_minus = 0
            for mm in self.Mgs:
                if mm in self.Mes:      Pi_minus += self.cg[(mm,0)] * self.Sv[self.ge_index[(mm,mm)]]
                if mm-1 in self.Mes:    Sigma_minus += 1/sqrt(2) * self.cg[(mm,-1)] * self.Sv[self.ge_index[(mm,mm-1)]]
                if mm+1 in self.Mes:    Sigma_minus += 1/sqrt(2) * self.cg[(mm,1)] * self.Sv[self.ge_index[(mm,mm+1)]]
            Pi_minus *= self.Ntotal
            Sigma_minus *= self.Ntotal
            Pi_plus = np.conj(Pi_minus)
            Sigma_plus = np.conj(Sigma_minus)
            
            out.append( np.sum(Pi_plus.real)/self.iterations )
            out.append( np.sum(Pi_plus.imag)/self.iterations )
            out.append( np.sum(Sigma_plus.real)/self.iterations )
            out.append( np.sum(Sigma_plus.imag)/self.iterations )
            
            #######################
            ###     Two-point   ###
            #######################
            
            if param.method in ['MF','TWA']:
                
                oneminus1overN = 1 - 1/self.Ntotal
        
                # Pi Pi
                temp = 0
                #for mm in self.Mgs:
                    #temp += 0.5 * self.cg[(mm,0)]**2 * ( self.Sv[self.ee_index[(mm,mm)]] - self.Sv[self.gg_index[(mm,mm)]] )
                for mm in self.Mgs:
                    temp += self.cg[(mm,0)]**2 * self.Sv[self.ee_index[(mm,mm)]]
                temp *= self.Ntotal
                out.append( np.sum((oneminus1overN*Pi_plus*Pi_minus + temp).real)/self.iterations )
        
                # Sigma Sigma
                temp = 0
                #for mm in self.Mgs:
                #    temp += 0.25 * ( self.cg[(mm,-1)]**2 * ( self.Sv[self.ee_index[(mm-1,mm-1)]] - self.Sv[self.gg_index[(mm,mm)]] ) \
                #                    + self.cg[(mm,1)]**2 * ( self.Sv[self.ee_index[(mm+1,mm+1)]] - self.Sv[self.gg_index[(mm,mm)]] ) \
                #                    + self.cg[(mm,-1)] * ( self.cg[(mm,1)]*self.Sv[self.ee_index[(mm-1,mm+1)]] - self.cg[(mm-2,1)]*self.Sv[self.gg_index[(mm-2,mm)]] ) \
                #                    + self.cg[(mm,1)] * ( self.cg[(mm,-1)]*self.Sv[self.ee_index[(mm+1,mm-1)]] - self.cg[(mm+2,-1)]*self.Sv[self.gg_index[(mm+2,mm)]] ) \
                #                    )
                for mm in self.Mgs:
                    temp += 0.5 * ( self.cg[(mm,-1)]**2 * self.Sv[self.ee_index[(mm-1,mm-1)]] \
                                    + self.cg[(mm,1)]**2 * self.Sv[self.ee_index[(mm+1,mm+1)]] \
                                    + self.cg[(mm,-1)] * self.cg[(mm,1)]*self.Sv[self.ee_index[(mm-1,mm+1)]] \
                                    + self.cg[(mm,1)] * self.cg[(mm,-1)]*self.Sv[self.ee_index[(mm+1,mm-1)]] \
                                    )
                temp *= self.Ntotal
                out.append( np.sum((oneminus1overN*Sigma_plus*Sigma_minus + temp).real)/self.iterations )
        
                # Pi Sigma
                temp = 0
                #for mm in self.Mgs:
                #    temp += 1/(2*sqrt(2)) * ( self.cg[(mm,0)] * ( self.cg[(mm,-1)]*self.Sv[self.ee_index[(mm,mm-1)]] - self.cg[(mm+1,-1)]*self.Sv[self.gg_index[(mm+1,mm)]] ) \
                #                            + self.cg[(mm,0)] * ( self.cg[(mm,1)]*self.Sv[self.ee_index[(mm,mm+1)]] - self.cg[(mm-1,1)]*self.Sv[self.gg_index[(mm-1,mm)]] ) \
                #                            )
                for mm in self.Mgs:
                    temp += 1/(sqrt(2)) * ( self.cg[(mm,0)] * self.cg[(mm,-1)]*self.Sv[self.ee_index[(mm,mm-1)]] \
                                            + self.cg[(mm,0)] * self.cg[(mm,1)]*self.Sv[self.ee_index[(mm,mm+1)]] \
                                            )
                temp *= self.Ntotal
                PiSigma = np.sum(oneminus1overN*Pi_plus*Sigma_minus + temp)/self.iterations
                out.append( 2*PiSigma.real )
                out.append( -2*PiSigma.imag )
            
            
            
            if param.method in ['cumulant','bbgky']:
            
                Sv = self.fill_auxiliary_variables(self.Sv)
            
                # Pi Pi
                PiPi = 0
                SigmaSigma = 0
                PiSigma = 0
                for mm in self.Mgs:
                    for nn in self.Mgs:
                        PiPi += self.cg[(mm,0)] * self.cg[(nn,0)] * Sv[self.egge_index[(mm,mm,nn,nn)]]
                        SigmaSigma += 0.5 * ( self.cg[(mm,-1)] * self.cg[(nn,-1)] * Sv[self.egge_index[(mm-1,mm,nn,nn-1)]] \
                                            + self.cg[(mm,-1)] * self.cg[(nn,1)] * Sv[self.egge_index[(mm-1,mm,nn,nn+1)]] \
                                            + self.cg[(mm,1)] * self.cg[(nn,-1)] * Sv[self.egge_index[(mm+1,mm,nn,nn-1)]] \
                                            + self.cg[(mm,1)] * self.cg[(nn,1)] * Sv[self.egge_index[(mm+1,mm,nn,nn+1)]] \
                                            )
                        PiSigma += 1/sqrt(2) * ( self.cg[(mm,0)] * self.cg[(nn,-1)] * Sv[self.egge_index[(mm,mm,nn,nn-1)]] \
                                                + self.cg[(mm,0)] * self.cg[(nn,1)] * Sv[self.egge_index[(mm,mm,nn,nn+1)]] \
                                                )
                PiPi *= self.Ntotal**2
                SigmaSigma *= self.Ntotal**2
                PiSigma *= self.Ntotal**2
                out.append( np.sum(PiPi.real)/self.iterations )
                out.append( np.sum(SigmaSigma.real)/self.iterations )
                out.append( np.sum(2*PiSigma.real)/self.iterations )
                out.append( np.sum(-2*PiSigma.imag)/self.iterations )
        
        
        
        
        if 'gell_mann' in param.which_observables:
        
            if param.method in ['MF','TWA']:
            
                # Complex gellmann_aa = |aa1><aa2| where aa1, aa2 are internal levels
                # Save expectation values of gellmann_aa
                for aa in range(self.localhilbertsize):
                    for bb in range(self.localhilbertsize):
                    
                        la = self.level_to_eg[aa]
                        lb = self.level_to_eg[bb]
                        ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                        mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                        
                        temp = np.sum( self.Sv[self.index[la+lb][(ma,mb)]] )/self.iterations
                        
                        out.append( temp.real )
                        out.append( temp.imag )
                
                # Save two-point functions of gellmann: 1/2 ( gellmann_aa * gellmann_bb + gellmann_bb * gellmann_aa )
                for aa in range(self.localhilbertsize):
                    for bb in range(self.localhilbertsize):
                        for cc in range(self.localhilbertsize):
                            for dd in range(self.localhilbertsize):
                    
                                la = self.level_to_eg[aa]
                                lb = self.level_to_eg[bb]
                                lc = self.level_to_eg[cc]
                                ld = self.level_to_eg[dd]
                                ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                                mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                                mc = self.Ms[lc][self.eg_to_level[lc].index(cc)]
                                md = self.Ms[ld][self.eg_to_level[ld].index(dd)]
                                
                                temp =  np.sum( self.Sv[self.index[la+lb][(ma,mb)]] * self.Sv[self.index[ld+lc][(md,mc)]] )/self.iterations
                                
                                out.append( temp.real )
                                out.append( temp.imag )
                
            
            if param.method in ['cumulant','bbgky']:
            
                Sv = self.fill_auxiliary_variables(self.Sv)
                
                # Complex gellmann_aa = |aa1><aa2| where aa1, aa2 are internal levels
                # Save expectation values of gellmann_aa
                for aa in range(self.localhilbertsize):
                    for bb in range(self.localhilbertsize):
                    
                        la = self.level_to_eg[aa]
                        lb = self.level_to_eg[bb]
                        ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                        mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                        
                        temp = np.sum( Sv[self.index[la+lb][(ma,mb)]] )/self.iterations
                        
                        out.append( temp.real )
                        out.append( temp.imag )
                
                
                # Save two-point functions of gellmann: 1/2 ( gellmann_aa * gellmann_bb^dagger + gellmann_bb^dagger * gellmann_aa )
                for aa in range(self.localhilbertsize):
                    for bb in range(self.localhilbertsize):
                        for cc in range(self.localhilbertsize):
                            for dd in range(self.localhilbertsize):
                    
                                la = self.level_to_eg[aa]
                                lb = self.level_to_eg[bb]
                                lc = self.level_to_eg[cc]
                                ld = self.level_to_eg[dd]
                                ma = self.Ms[la][self.eg_to_level[la].index(aa)]
                                mb = self.Ms[lb][self.eg_to_level[lb].index(bb)]
                                mc = self.Ms[lc][self.eg_to_level[lc].index(cc)]
                                md = self.Ms[ld][self.eg_to_level[ld].index(dd)]
                                
                                temp = 0.5 * ( np.sum( Sv[self.index[la+lb+ld+lc][(ma,mb,md,mc)]] )/self.iterations \
                                               + np.sum( Sv[self.index[ld+lc+la+lb][(md,mc,ma,mb)]] )/self.iterations )
                                
                                out.append( temp.real )
                                out.append( temp.imag )
            
            
            
        if 'fisher' in param.which_observables:
            print('ERROR: fisher not implemented for MF/TWA/cumulant/BBGKY.')
        
        
        return out
        
        
        
        
        
    def read_fullDistribution (self):
        """
        Outputs total occupation of each level of each iteration.
        """
        out = []
        
        if param.method in ['TWA','bbgky']:
            
            # Read populations
            pops = []
            for aa in self.Mgs:
                pops.append( self.Sv[self.gg_index[(aa,aa)]].real )
            for bb in self.Mes:
                pops.append( self.Sv[self.ee_index[(bb,bb)]].real )
                
            excitations = np.sum( np.array(pops[param.deg_g:]), axis=0 )
            
            ## Prepare Binning
            bins = int(np.max([self.iterations/param.bin_factor,10]))
            step = 1/bins
            nlevels = param.deg_g+param.deg_e
            binned_pops = np.zeros((bins,1+nlevels))
            binned_excit = np.zeros((bins,1))
            
            # Check negative populations
            unphysical_trajs = np.zeros(len(pops))
            epsilon = 5/self.Ntotal
            for ss in range(len(pops)):
                if (pops[ss]+epsilon<0).any() or (pops[ss]>1).any():
                    unphysical_trajs[ss] = np.sum(pops[ss]+epsilon<0) + np.sum((pops[ss]>1))
                    print('WARNING: Negative populations found in %i trajectories for %i level. They will not count for fullDistribution.'%(unphysical_trajs[ss],ss))
            unphysical_trajs_ee = 0
            if (excitations+epsilon<0).any() or (excitations>1).any():
                unphysical_trajs_ee = np.sum(excitations+epsilon<0) + np.sum((excitations>1))
                print('WARNING: Negative populations found in %i trajectories for total n_e. They will not count for fullDistribution.'%(unphysical_trajs_ee))
            
            # Binning
            binL=0
            for ii in range(bins):
                
                #binL = ii*step
                binR = binL+step  # (ii+1)*step
                
                binned_pops[ii,0] = binL
                
                for ss in range(len(pops)):
                    binned_pops[ii,1+ss] = np.sum( ( epsilon*(min(1,ii)-1)+binL<=pops[ss]) * (pops[ss]<binR) )
                    
                binned_excit[ii,0] = np.sum( (epsilon*(min(1,ii)-1)+binL<=excitations) * (excitations<binR) )
                    
                binL = binR
            
            binned_pops[:,1:] *= 1/( self.iterations-unphysical_trajs.reshape((1,len(unphysical_trajs))) )
            binned_excit *= 1/( self.iterations-unphysical_trajs_ee )
            
            #print(binned_pops)
            #print(binned_excit)
            
            #print(np.concatenate((binned_pops,binned_excit),axis=1))
            
            #out.append( binned_excit )
            
            """
            excitations = np.array([ self.Sv[self.ee_index[(bb,bb)]] for bb in self.Mes ], axis=0)
            
            ## Binning
            bins = np.min([self.iterations/10,10])
            step = 1/bins
            binned_excit = np.zeros((bins,2))
            
            #unphysical_trajs = 0
            #if (excitations<0).any() or (excitations>1).any():
            #    unphysical_trajs = np.sum(excitations<0) + np.sum((excitations>1))
            #    print('WARNING: Negative populations found in %i trajectories. They will not count for fullDistribution.'%(unphysical_trajs))
            
            for ii in range(bins):
                
                binL = ii*step
                binR = (ii+1)*step
                
                binned_excit[ii,0] = binL
                binned_excit[ii,1] = np.sum( (binL<=excitations) * (excitations<binR) )
            
            binned_excit[:,1] *= 1/(self.iterations-unphysical_trajs)
            
            out.append( binned_excit )
            
            #out.append( np.array([ self.Sv[self.ee_index[(bb,bb)]] for bb in self.Mes ], axis=0) )
            
            """
            
        else:
            print('ERROR: fullDistribution only implemented for ED or TWA.')
        
        return np.concatenate((binned_pops,binned_excit),axis=1)

        













