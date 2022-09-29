
import math
from math import sqrt as sqrt

import numpy as np
from numpy.linalg import eig
from numpy.linalg import eigh
from numpy.linalg import eigvalsh
from numpy.linalg import norm as normLA
from numpy import sin as sin
from numpy import cos as cos
from numpy import exp as exp

import os

import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.special import comb
from scipy.integrate import complex_ode

from sympy.physics.quantum.cg import CG
from sympy import S

from scipy.stats import unitary_group

from collections import defaultdict

import sys

import parameters as param
import hilbert_space

# Contains lattice parameters and functions for indexing

class Cavity_ED:
    
    """
    Class parameters: (self)
    
    
    """
    
    
    """
    Class functions:
    
    
    """
    
    def __init__(self):
        
        
        ### ---------------
        ###     GEOMETRY
        ### ---------------
        self.dim = 0
        self.Nlist = []
        
        if param.geometry=='alltoall':
            self.Nlist.append(param.Ntotal);
            self.dim = 1;
        else: print("\ncavity_ED/init: Wrong type of geometry chosen.\n")
    
        self.Ntotal=1
        for ii in range(self.dim): self.Ntotal *= self.Nlist[ii];
        
        
        ### ---------------
        ###     CONSTANTS
        ### ---------------
        
        self.dummy_constants()
        
        
        ### ---------------
        ###     LEVELS
        ### ---------------
        self.levels_info = { 'Fe':param.Fe, 'deg_e':param.deg_e, 'Fg':param.Fg, 'deg_g':param.deg_g }
        #self.Mgs = [ -param.Fg+mm for mm in range(param.deg_g) ]
        #self.Mes = [ -param.Fe+mm for mm in range(param.deg_e) ]
        self.Mgs = [ -param.Fg+param.start_g+mm for mm in range(param.deg_g) ]
        self.Mes = [ -param.Fe+param.start_e+mm for mm in range(param.deg_e) ]
        self.Qs = [0,-1,1]
        #[ expression for item in list if conditional ]
        
        
        
        ### ---------------
        ###  SPACE VECTORS
        ### ---------------
        self.r_i = [np.zeros(3) for _ in range(self.Ntotal) ]
        self.fill_ri()
        
        
        
        ### -------------------------
        ###    CLEBSCH-GORDAN COEFF
        ### -------------------------
        self.cg = {}
        for mg in self.Mgs:
            for q in self.Qs:
                if mg+q in self.Mes and q not in param.clebsch_zero_q:
                    #self.cg[(mg,q)] = float(CG(param.Fg, mg, 1, q, param.Fe, mg+q).doit())
                    self.cg[(mg,q)] = CG(param.Fg, mg, 1, q, param.Fe, mg+q).doit()
        if param.Fg==0 and param.Fe==0: self.cg[(0,0)] = 1.0    # Two-level case
        self.cg = defaultdict( lambda: 0, self.cg )     # Set all other Clebsch-Gordan coefficients to zero
        
        print(self.cg)
        
        
        
        
        ### ---------------
        ###  HILBERT SPACE
        ### ---------------
        # Mapping between e-g numbering and a flattened-out level numbering, which will be used for Hilbert space.
        # Ordering of atomic levels: g to e, left to right
        self.eg_to_level = { 'g': [ bb for bb in range(param.deg_g) ] ,\
                             'e': [ param.deg_g+aa for aa in range(param.deg_e) ] }
        self.level_to_eg = []
        for nn in range(self.numlevels):
            if nn<param.deg_g: self.level_to_eg.append(['g',nn])
            if nn>=param.deg_g: self.level_to_eg.append(['e',nn-param.deg_g])
         
        self.hspace = hilbert_space.Hilbert_Space(self.numlevels,param.filling,self.Ntotal,param.hilberttype)
        self.Pi_Sigma_defined = False
        
        
        if param.hilberttype=='sym':
            
            if param.output_dark_renyi or param.output_purity_renyi or param.output_squeezing or ('gell_mann' in param.which_observables) or ('fisher' in param.which_observables):
            
                self.define_Jxyz_collective_sym()
                self.define_gellmann_collective_sym()
                self.define_complex_gellmann_collective_sym()
        
        
        ### ---------------
        ###     MEMORY
        ### ---------------
        # Real
        self.memory_full_Linblad = self.hspace.hilbertsize**4 * 2 * 8 / 1024**3        # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
        self.memory_full_Hamiltonian = self.hspace.hilbertsize**2 * 2 * 8 / 1024**3        # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
        self.memory_wavefunction = self.hspace.hilbertsize * 2 * 8 / 1024**3        # Number of entries x 2 doubles/complex number x 8 bytes/double  /  Bytes per Gb
        print("\nMemory full Linblad: %g Gb."%(self.memory_full_Linblad))
        print("Memory full Hamiltonian: %g Gb."%(self.memory_full_Hamiltonian))
        print("Memory wave-function: %g Gb."%(self.memory_wavefunction))
        
        # Estimated
        if param.hilberttype=='full':
            self.memory_estimate_sparse_Linblad = 15 * self.Ntotal**2 * (min([param.deg_e,param.deg_g]))**2 \
                                                 * (self.hspace.hilbertsize/self.hspace.localhilbertsize)**2 * 4 * 8 /1024**3
            # Number of PiPi and SigmaSigma terms x Number of terms in sum over {ii,jj,mm,nn} x nonzero entries in sigma_alphabeta^ii x tensor product with "1" x (2 position integers + 1 complex number) x 8 bytes/double / Bytes per Gb
            self.memory_estimate_sparse_Hamiltonian = 5 * self.Ntotal**2 * (min([param.deg_e,param.deg_g]))**2 \
                                                 * (self.hspace.hilbertsize/self.hspace.localhilbertsize**2) * 4 * 8 /1024**3
        if param.hilberttype=='sym':
            minge = min([param.deg_e,param.deg_g])
            dlt = self.kron_del(param.deg_e,param.deg_g)
            frac1 = 1 - (self.numlevels-1)/(self.Ntotal+self.numlevels-1)
            frac2 = 1 - (self.numlevels-1)/(self.Ntotal+self.numlevels-1)*( 2 - (self.numlevels-2)/(self.Ntotal+self.numlevels-2) )
            
            #self.memory_estimate_sparse_Hamiltonian = 5 * (min([param.deg_e,param.deg_g]))**2 * (self.hspace.hilbertsize) * 4 * 8 /1024**3
            #self.memory_estimate_sparse_Hamiltonian = ( minge*(minge-1) + 2*(minge-dlt)*(2*(minge-dlt)-1) + 1 ) * (self.hspace.hilbertsize) * (4+4+2*8) #/1024**3
            ## Much tighter estimate based on cavity Hamiltonian only:
            self.memory_estimate_sparse_Hamiltonian = ( frac2*( minge*(minge-1) + 2*(2*(minge-dlt)**2 - (minge-dlt) - (param.deg_e-2) ) ) + frac1*2*(param.deg_e-2) + 1 ) * (self.hspace.hilbertsize) * (4+4+2*8) /1024**3
            
            #self.memory_estimate_sparse_Linblad = 15 * (min([param.deg_e,param.deg_g]))**2 * (self.hspace.hilbertsize)**2 * 4 * 8 /1024**3
            self.memory_estimate_sparse_Linblad = 2 * self.hspace.hilbertsize * self.memory_estimate_sparse_Hamiltonian + ( minge**2 + 4*(minge-dlt)**2 ) * (frac1*self.hspace.hilbertsize)**2 * (4+4+2*8) /1024**3
            # Number of PiPi and SigmaSigma terms x Number of terms in sum over {mm,nn} x nonzero entries in Pi_m x (2 position integers + 1 complex number) x 8 bytes/double / Bytes per Gb
        
        print("Estimated memory of sparse Linblad: %g Gb."%(self.memory_estimate_sparse_Linblad))
        print("Estimated memory of sparse Hamiltonian: %g Gb."%(self.memory_estimate_sparse_Hamiltonian))
        
        
        
        
        
    def dummy_constants (self):
        
        self.numlevels = param.deg_e + param.deg_g      # Total number of internal electronic states per atom
        
        
    def fill_ri (self):
        """Fill out matrix of atom positions r_i"""
        #if (param.geometry=='alltoall'):
            # Do nothing
            
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
       
    def kron_del(self,a1,a2):
        """Kroenecker delta"""
        if a1==a2:
            return 1
        else:
            return 0
        
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

    

####################################################################

#######                INITIAL CONDITIONS                ###########

####################################################################

    def choose_initial_condition (self,cIC=param.cIC):
        """Initialize density matrix to chosen initial condition."""
        optionsIC = defaultdict( self.IC_initialstate )
        optionsIC = { 'initialstate': self.IC_initialstate,
                        'mixedstate': self.IC_mixedstate,
                        'puresupstate': self.IC_puresupstate,
                        'byhand': self.IC_byhand }
        optionsIC[cIC]()
        
        self.rhoV = self.rho.flatten('C').reshape(self.hspace.hilbertsize**2,1)        # 'C': row-major, "F": column-major
        
        print(self.rho)
        #print(self.hspace.state_to_index)
        
        # Check initial trace
        if abs(np.trace(self.rho)-1)>0.000000001: print("\nWARNING/choose_initial_condition: Trace of rho is initially not 1.\n")
        print("Initial trace of rho is %g."%(np.trace(self.rho)))
        
        # Perform rotation
        if param.rotate==True:
            self.rotate_IC()
            print("Trace after rotation is ",np.trace(self.rho),".")
        
        
        print(self.rho)
        
        
    def IC_initialstate (self):
        """
        Initialize all sites with the atoms in the state specified by param.initialstate.
        """
        if len(param.initialstate)!=param.filling: print("\nWarning: Length of initialstate doesn't match filling.\n")
        
        occlevels = []
        for ii in range(len(param.initialstate)):
            occlevels.append( self.eg_to_level[ param.initialstate[ii][0] ][ int(param.initialstate[ii][1:]) ] )    
        occupied_localstate = self.hspace.get_statenumber( occlevels )
        
        if param.hilberttype=='full':
            rhoi = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize))
            rhoi[occupied_localstate,occupied_localstate] = 1
            self.rho = rhoi
            for ii in range(1,self.Ntotal):
                self.rho = np.kron( self.rho , rhoi )
            
        if param.hilberttype=='sym':
            occupied_symstate = [ 0 for ii in range(self.numlevels)]
            occupied_symstate[occupied_localstate] = self.Ntotal
            occupied_symstate_idx = self.hspace.state_to_index[tuple(occupied_symstate)]
            
            self.rho = np.zeros((self.hspace.hilbertsize,self.hspace.hilbertsize))
            self.rho[occupied_symstate_idx,occupied_symstate_idx] = 1
            
            # Maybe use projectors of hilbert_space to do this? Use sparse array for rho?
            
        
    def IC_mixedstate(self):
        """Initialize density matrix with each atom in a mixed state of ground states"""
        
        if param.hilberttype=='full':
            rhoi = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize))
            #for mg in range(param.deg_g): rhoi[mg,mg] = param.mixed_gs_probabilities[mg]
            for mg in range(param.deg_g): rhoi[self.eg_to_level['g'][mg],self.eg_to_level['g'][mg]] = param.mixed_gs_probabilities[mg]
            for me in range(param.deg_e): rhoi[self.eg_to_level['e'][me],self.eg_to_level['e'][me]] = param.mixed_es_probabilities[me]
            self.rho = rhoi
            for ii in range(1,self.Ntotal):
                self.rho = np.kron( self.rho , rhoi )
        
        if param.hilberttype=='sym':
            self.rho = np.zeros((self.hspace.hilbertsize,self.hspace.hilbertsize))
            
            for state in self.hspace.state_to_index:
                probabilities_prod = np.prod( [ param.mixed_gs_probabilities[mg]**state[self.eg_to_level['g'][mg]] for mg in range(param.deg_g) ] ) \
                                    * np.prod( [ param.mixed_es_probabilities[me]**state[self.eg_to_level['e'][me]] for me in range(param.deg_e) ] )
                self.rho[ self.hspace.state_to_index[tuple(state)] , self.hspace.state_to_index[tuple(state)] ] += self.multinomial(list(state)) * probabilities_prod
            
            #for state in self.hspace.state_to_index:        ### Not very efficient to run over all states, but well
            #    state_gslice = state[:param.deg_g]
            #    if sum(state[param.deg_g:])==0:     # if no atom in e state
            #        probabilities_prod = np.prod( [ param.mixed_gs_probabilities[mg]**state[mg] for mg in range(param.deg_g) ] )
            #        self.rho[ self.hspace.state_to_index[tuple(state)] , self.hspace.state_to_index[tuple(state)] ] += self.multinomial(list(state_gslice)) * probabilities_prod
                    
    
    def IC_puresupstate(self):
        """Initialize density matrix with each atom in a pure superposition state of ground levels"""
        
        if param.hilberttype=='full':
            psi_i = np.zeros((self.hspace.localhilbertsize,1))
            for mg in range(param.deg_g): psi_i[mg,0] = param.pure_gs_amplitudes[mg]
            for me in range(param.deg_e): psi_i[self.eg_to_level['e'][me],0] = param.pure_es_amplitudes[me]
            rhoi = psi_i @ psi_i.conj().T
            self.rho = rhoi
            for ii in range(1,self.Ntotal):
                self.rho = np.kron( self.rho , rhoi )
        
        if param.hilberttype=='sym':
            psi = np.zeros((self.hspace.hilbertsize,1))
            for state in self.hspace.state_to_index:        ### Not very efficient to run over all states, but well
                amplitudes_prod = np.prod( [ param.pure_gs_amplitudes[mg]**state[self.eg_to_level['g'][mg]] for mg in range(param.deg_g) ] ) \
                                 * np.prod( [ param.pure_es_amplitudes[me]**state[self.eg_to_level['e'][me]] for me in range(param.deg_e) ] )
                psi[ self.hspace.state_to_index[tuple(state)] , 0 ] += sqrt( self.multinomial(list(state)) ) * amplitudes_prod
            
                #if sum(state[param.deg_g:])==0:     # if no atom in e state
                #    amplitudes_prod = np.prod( [ param.pure_gs_amplitudes[mg]**state[mg] for mg in range(param.deg_g) ] )
                #    psi[ self.hspace.state_to_index[tuple(state)] , 0 ] += sqrt( self.multinomial(list( state[:param.deg_g] )) ) * amplitudes_prod
            
            self.rho = psi @ psi.conj().T
            
            
            
            
    def IC_byhand (self):
        """ Initialize density matrix of one atom in some state hardcoded by hand """
        
        if param.hilberttype=='full':
        
            # |g_{1/2}><g_{1/2}| + superpos of g_{-1/2} and e_{-1/2}
            rhoi = np.zeros((self.hspace.localhilbertsize,self.hspace.localhilbertsize), dtype=complex)
            occupied_localstate = self.eg_to_level[ 'g' ][ 1 ]
            rhoi[occupied_localstate,occupied_localstate] = 0.5
        
            psi_i = np.array( [[1/sqrt(2)], [0], [1/sqrt(2)], [0]] , dtype=complex)
            rhoi += 0.5 * psi_i @ psi_i.conj().T
        
            self.rho = rhoi
            for ii in range(1,self.Ntotal):
                self.rho = np.kron( self.rho , rhoi )
        
        if param.hilberttype=='sym':
            print('\n/ERROR:IC_byhand: This type of IC not implemented for sym Hilbert space.\n')
        
        
    def rotate_IC(self):
        """Rotate initial density matrix with H_Rabi for time t=1."""
        if self.Pi_Sigma_defined==False: self.define_Pi_Sigma()
        
        # Define Rabi Hamiltonian and put in superoperator form
        one = sp.identity(self.hspace.hilbertsize,format='csc')
        H_rabi_IC = param.Omet_Pi * self.Pi_minus + param.Omet_Sigma * self.Sigma_minus
        H_rabi_IC = H_rabi_IC + H_rabi_IC.conj().T
        
        linSuper_IC =  - 1j * ( sp.kron(H_rabi_IC,one,format='csc') - sp.kron(one,H_rabi_IC.conj(),format='csc') )
        
        # Define Linblad equation
        def linblad_equation_IC(t,psi):
            return linSuper_IC @ psi
        
        # Set solver
        ICsolver = complex_ode(linblad_equation_IC).set_integrator('dopri5',atol=param.atol,rtol=param.rtol,nsteps=2000)
        ICsolver.set_initial_value(self.rhoV.reshape((len(self.rhoV))), 0)
        
        print('atol:', ICsolver._integrator.atol)
        print('rtol:', ICsolver._integrator.rtol)
        
        # Evolve for t=1
        if ICsolver.successful():
            ICsolver.integrate(ICsolver.t+1)
            self.rho = ICsolver.y.reshape(self.hspace.hilbertsize,self.hspace.hilbertsize)
            self.rhoV = self.rho.flatten('C').reshape(self.hspace.hilbertsize**2,1)
        else: print("\nERROR/rotate_IC: Problem with ICsolver, returns unsuccessful.\n")
        
        



####################################################################

###########                SECOND PHASE                #############

####################################################################

    def rotate_pulse(self,Omet_Pi_pulse,Omet_Sigma_pulse):
        """Rotate density matrix with H_Rabi for time t=1. Equivalent to a quick pulse."""
        if self.Pi_Sigma_defined==False: self.define_Pi_Sigma()
        
        # Define Rabi Hamiltonian and put in superoperator form
        one = sp.identity(self.hspace.hilbertsize,format='csc')
        H_rabi_pulse = Omet_Pi_pulse * self.Pi_minus + Omet_Sigma_pulse * self.Sigma_minus
        H_rabi_pulse = H_rabi_pulse + H_rabi_pulse.conj().T
        
        linSuper_pulse =  - 1j * ( sp.kron(H_rabi_pulse,one,format='csc') - sp.kron(one,H_rabi_pulse.conj(),format='csc') )
        
        # Define Linblad equation
        def linblad_equation_pulse(t,psi):
            return linSuper_pulse @ psi
        
        # Set solver
        pulse_solver = complex_ode(linblad_equation_pulse).set_integrator('dopri5',atol=param.atol,rtol=param.rtol,nsteps=2000)
        pulse_solver.set_initial_value(self.rhoV.reshape((len(self.rhoV))), 0)
        
        print('atol:', pulse_solver._integrator.atol)
        print('rtol:', pulse_solver._integrator.rtol)
        
        # Evolve for t=1
        if pulse_solver.successful():
            pulse_solver.integrate(pulse_solver.t+1)
            self.rho = pulse_solver.y.reshape(self.hspace.hilbertsize,self.hspace.hilbertsize)
            self.rhoV = self.rho.flatten('C').reshape(self.hspace.hilbertsize**2,1)
        else: print("\nERROR/rotate_system: Problem with pulse_solver, returns unsuccessful.\n")

    def redefine_H_rabi (self,rabi_Pi_new,rabi_Sigma_new):
        """Saves Hamiltonian of Rabi coupling, only first summand."""
        self.H_rabi = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        
        self.H_rabi = rabi_Pi_new * self.Pi_minus + rabi_Sigma_new * self.Sigma_minus
        
        self.H_rabi = self.H_rabi + self.H_rabi.conj().T
        
    
        
        
####################################################################

######                LINBLAD SUPEROPERATOR                #########

####################################################################



    def save_HamLin (self):
        """
        Saves separately the Hamiltonians and Linbladians of the dipolar, rabi and energies parts.
        """
        if self.Pi_Sigma_defined==False: self.define_Pi_Sigma()
        
        self.define_H_cavity()
        self.define_L_cavity()
        self.define_H_rabi()
        
        if param.hilberttype=='full':
            self.define_H_energies()
        if param.hilberttype=='sym':
            self.define_H_energies_sym()
    
    
    def define_linblad_superop (self,phase):
        """
        Computes Linblad superoperator and effective Hamiltonian.
        For time-dependent problems it saves Linblads to speed up computation of time-dependent Linblad during dynamics.
        """
        one = sp.identity(self.hspace.hilbertsize,format='csc')
        self.linSuper = csc_matrix( (self.hspace.hilbertsize**2,self.hspace.hilbertsize**2) )
        self.ham_eff_total = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        # Energies, cavity Hamiltonians
        self.ham_eff_total = self.ham_eff_cavity + self.H_energies + self.H_rabi
                
        # Total Linblad (time-independent part)
        self.linSuper = self.linSuper_cavity - 1j * ( sp.kron(self.ham_eff_total,one,format='csc') - sp.kron(one,self.ham_eff_total.conj(),format='csc') )


    def compute_memory (self):
        """Compute memory and sparsity of Linblad and Hamiltonian"""
        self.memory_sparse_Linblad = (self.linSuper.data.nbytes + self.linSuper.indptr.nbytes + self.linSuper.indices.nbytes) / 1024**3        # Number of Bytes  /  Bytes per Gb
        self.memory_sparse_Hamiltonian = (self.ham_eff_total.data.nbytes + self.ham_eff_total.indptr.nbytes + self.ham_eff_total.indices.nbytes) / 1024**3        # Number of Bytes  /  Bytes per Gb
        
        self.memory_sparse_cavityHamiltonian = (self.ham_eff_cavity.data.nbytes + self.ham_eff_cavity.indptr.nbytes + self.ham_eff_cavity.indices.nbytes) / 1024**3
        
        #self.memory_sparse_Hamiltonian = (self.ham_eff_cavity.data.nbytes + self.ham_eff_cavity.indptr.nbytes + self.ham_eff_cavity.indices.nbytes) / 1024**3        # Number of Bytes  /  Bytes per Gb
        #print("\nCavity Ham: indices: %g bytes"%(self.ham_eff_total.indices.nbytes))
        #print("Cavity Ham: indptr: %g bytes"%(self.ham_eff_total.indptr.nbytes))
        #print("Cavity Ham: data: %g bytes"%(self.ham_eff_total.data.nbytes))
        #print("Cavity Ham: total: %g bytes"%(self.ham_eff_total.data.nbytes + self.ham_eff_total.indptr.nbytes + self.ham_eff_total.indices.nbytes))
        
        #print("\nCavity Lin: indices: %g bytes"%(self.linSuper.indices.nbytes))
        #print("Cavity Lin: indptr: %g bytes"%(self.linSuper.indptr.nbytes))
        #print("Cavity Lin: data: %g bytes"%(self.linSuper.data.nbytes))
        #print("Cavity Lin: total: %g bytes"%(self.linSuper.data.nbytes + self.linSuper.indptr.nbytes + self.linSuper.indices.nbytes))
        
        if self.memory_sparse_cavityHamiltonian>self.memory_estimate_sparse_Hamiltonian: print("\nWARNING: Memory estimate of Hamiltonian below real memory of Hamiltonian.\n")
        
        self.sparsity_Linblad = self.linSuper.nnz / np.prod(self.linSuper.shape)
        self.sparsity_Hamiltonian = self.ham_eff_total.nnz / np.prod(self.ham_eff_total.shape)
        print("\nMemory for sparse Linblad: %g Gb."%(self.memory_sparse_Linblad))
        print("Memory for sparse Hamiltonian: %g Gb."%(self.memory_sparse_Hamiltonian))
        print("Sparsity of Linblad: %g"%(self.sparsity_Linblad))
        print("Sparsity of Hamiltonian: %g"%(self.sparsity_Hamiltonian))
        
        #sys.getsizeof(self.ham_eff_total)
        
    
    def define_Pi_Sigma(self):
        """ Define and save operators Pi and Sigma """
        if param.hilberttype=='full':
            self.define_Pi_Sigma_full()
        if param.hilberttype=='sym':
            self.define_Pi_Sigma_sym()
            
        self.Pi_Sigma_defined = True
    
    
    def define_Pi_Sigma_sym(self):
        """ Define and save operators Pi and Sigma for symmetric manifold """
        # Define Pi and Sigma operators (it's faster to build in lil_matrix and then convert to csc)
        self.Pi_minus = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        self.Sigma_minus = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        # Fill Pi and Sigma operators
        for mm in self.Mgs:
            mgind = self.eg_to_level['g'][self.Mgs.index(mm)]
            
            # Pi
            if mm in self.Mes:
                meind = self.eg_to_level['e'][self.Mes.index(mm)]
                #cg = float(CG(param.Fg, mm, 1, 0, param.Fe, mm).doit())
                cg = self.cg[(mm,0)]
                changestate = np.zeros(self.numlevels, dtype='int')
                changestate[mgind] = 1
                changestate[meind] = -1
                
                for state in self.hspace.state_to_index:
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate + changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        self.Pi_minus[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += cg * sqrt( rstate[meind] * (rstate[mgind]+1) )
                        #### This is considerably slower:
                        #self.Pi_minus += cg * sqrt( rstate[meind] * (rstate[mgind]+1) ) * self.hspace.transitionOp_sym(lstate,rstate)
                        
            # Sigma -1
            if mm-1 in self.Mes:
                meind = self.eg_to_level['e'][self.Mes.index(mm-1)]
                #cg = float(CG(param.Fg, mm, 1, -1, param.Fe, mm-1).doit())
                cg = self.cg[(mm,-1)]
                changestate = np.zeros(self.numlevels, dtype='int')
                changestate[mgind] = 1
                changestate[meind] = -1
                
                for state in self.hspace.state_to_index:
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate + changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        self.Sigma_minus[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += 1.0/sqrt(2) * cg * sqrt( rstate[meind] * (rstate[mgind]+1) )
                        #self.Sigma_minus += cg * sqrt( rstate[meind] * (rstate[mgind]+1) ) * self.hspace.transitionOp_sym(lstate,rstate)
                
            
            # Sigma +1
            if mm+1 in self.Mes:
                meind = self.eg_to_level['e'][self.Mes.index(mm+1)]
                #cg = float(CG(param.Fg, mm, 1, 1, param.Fe, mm+1).doit())
                cg = self.cg[(mm,1)]
                changestate = np.zeros(self.numlevels, dtype='int')
                changestate[mgind] = 1
                changestate[meind] = -1
                
                for state in self.hspace.state_to_index:
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate + changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        self.Sigma_minus[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += 1.0/sqrt(2) * cg * sqrt( rstate[meind] * (rstate[mgind]+1) )
                        #self.Sigma_minus += cg * sqrt( rstate[meind] * (rstate[mgind]+1) ) * self.hspace.transitionOp_sym(lstate,rstate)
        
        self.Pi_minus.tocsc()
        self.Sigma_minus.tocsc()
        
        # Plus operators
        self.Pi_plus = self.Pi_minus.conj().T
        self.Sigma_plus = self.Sigma_minus.conj().T
        
        #print(self.Pi_plus)
        #print(np.sqrt(2)*self.Sigma_plus)
    
    
    def define_Pi_Sigma_full(self):
        """ Define and save operators Pi and Sigma for full Hilbert space """
        """CHANGE CONSTRUCTION OF PI and SIGMA TO LIL_MATRIX"""
        # Define Pi and Sigma operators
        self.Pi_minus = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        self.Sigma_minus = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        # Fill Pi and Sigma operators
        for ii in range(self.Ntotal):
            for mm in self.Mgs:
                mgind = self.Mgs.index(mm)
                
                if mm in self.Mes:
                    meind = self.Mes.index(mm)
                    sigma_ge = self.hspace.sigma_matrix( self.eg_to_level['g'][mgind] , self.eg_to_level['e'][meind] , ii )
                    #cg = float(CG(param.Fg, mm, 1, 0, param.Fe, mm).doit())
                    cg = self.cg[(mm,0)]
                    self.Pi_minus = self.Pi_minus + cg * sigma_ge
                
                
                if mm-1 in self.Mes:
                    meind = self.Mes.index(mm-1)
                    sigma_ge = self.hspace.sigma_matrix( self.eg_to_level['g'][mgind] , self.eg_to_level['e'][meind] , ii )
                    #cg = float(CG(param.Fg, mm, 1, -1, param.Fe, mm-1).doit())
                    cg = self.cg[(mm,-1)]
                    self.Sigma_minus = self.Sigma_minus + 1.0/sqrt(2) * cg * sigma_ge
                    
                if mm+1 in self.Mes:
                    meind = self.Mes.index(mm+1)
                    sigma_ge = self.hspace.sigma_matrix( self.eg_to_level['g'][mgind] , self.eg_to_level['e'][meind] , ii )
                    #cg = float(CG(param.Fg, mm, 1, 1, param.Fe, mm+1).doit())
                    cg = self.cg[(mm,1)]
                    self.Sigma_minus = self.Sigma_minus + 1.0/sqrt(2) * cg * sigma_ge
                    
        # Plus operators
        self.Pi_plus = self.Pi_minus.conj().T
        self.Sigma_plus = self.Sigma_minus.conj().T

    
    def define_H_cavity (self):
        """Saves Hamiltonian part of cavity-mediated interactions."""
        self.ham_eff_cavity = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
                
        # Compute Hamiltonian
        self.ham_eff_cavity = (param.g_coh - 1j*param.g_inc/2) * ( self.Pi_plus@self.Pi_minus + self.Sigma_plus@self.Sigma_minus )

    
    def define_L_cavity (self):
        """Saves Linbladian part of cavity-mediated interactions."""
        self.linSuper_cavity = csc_matrix( (self.hspace.hilbertsize**2,self.hspace.hilbertsize**2) )
        
        # Compute Linblad
        self.linSuper_cavity = param.g_inc * ( sp.kron( self.Pi_minus , self.Pi_plus.T , format='csc' ) + sp.kron( self.Sigma_minus , self.Sigma_plus.T , format='csc' ) )
        
    
    def define_H_rabi (self):
        """Saves Hamiltonian of Rabi coupling, only first summand."""
        self.H_rabi = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        
        self.H_rabi = param.rabi_Pi * self.Pi_minus + param.rabi_Sigma * self.Sigma_minus
        
        self.H_rabi = self.H_rabi + self.H_rabi.conj().T
    

    def define_H_energies_sym (self):
        """Saves Hamiltonian of energy levels including Zeeman shifts. In rotating frame of omegaR."""
        self.H_energies = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        # Fill H_energies for mg
        for mm in self.Mgs:
            mgind = self.eg_to_level['g'][self.Mgs.index(mm)]
            prefac = float(mm)*param.zeeman_g
            
            for state in self.hspace.state_to_index:
                self.H_energies[ self.hspace.state_to_index[tuple(state)] , self.hspace.state_to_index[tuple(state)] ] += prefac * state[mgind]
                
        # Fill H_energies for me
        for mm in self.Mes:
            meind = self.eg_to_level['e'][self.Mes.index(mm)]
            prefac = float(mm)*param.zeeman_e - param.detuning + param.epsilon_ne
            
            for state in self.hspace.state_to_index:
                self.H_energies[ self.hspace.state_to_index[tuple(state)] , self.hspace.state_to_index[tuple(state)] ] += prefac * state[meind]
                
        
        self.H_energies.tocsc()
        #print(self.H_energies)
        
        # Add extra term to break possible degeneracies of different n_e states for eigenstate calculation
        #if param.epsilon_ne!=0:
        #    for mm in self.Mes:
        #        meind = self.eg_to_level['e'][self.Mes.index(mm)]
        #        for state in self.hspace.state_to_index:
        #            self.H_energies[ self.hspace.state_to_index[tuple(state)] , self.hspace.state_to_index[tuple(state)] ] += param.epsilon_ne * state[meind]
        
        return self.H_energies

    
    def define_H_energies (self):
        """Saves Hamiltonian of energy levels including Zeeman shifts. In rotating frame of omegaR."""
        self.H_energies = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        for ii in range(self.Ntotal):
            for aa in range(param.deg_e):
                sigma_aa_i = self.hspace.sigma_matrix( self.eg_to_level['e'][aa] , self.eg_to_level['e'][aa] , ii )
                #self.H_energies = self.H_energies + ((param.omega0+(-param.Fe+aa)*param.zeeman_e)-param.omegaR) * sigma_aa_i
                self.H_energies = self.H_energies + ( float(self.Mes[aa])*param.zeeman_e - param.detuning + param.epsilon_ne ) * sigma_aa_i
                
            for bb in range(param.deg_g):
                sigma_bb_i = self.hspace.sigma_matrix( self.eg_to_level['g'][bb] , self.eg_to_level['g'][bb] , ii )
                #self.H_energies = self.H_energies + (bb*param.zeeman_g) * sigma_bb_i
                self.H_energies = self.H_energies + float(self.Mgs[bb])*param.zeeman_g * sigma_bb_i
        
        # Add extra term to break possible degeneracies of different n_e states for eigenstate calculation
        #if param.epsilon_ne!=0:
        #    for ii in range(self.Ntotal):
        #        for aa in range(param.deg_e):
        #            sigma_aa_i = self.hspace.sigma_matrix( self.eg_to_level['e'][aa] , self.eg_to_level['e'][aa] , ii )
        #            self.H_energies = self.H_energies + param.epsilon_ne * sigma_aa_i
        
        # Energy shift of n_e=2 
        #if param.epsilon_ne2!=0:
        #    for ii in range(self.Ntotal):
        #        projection_i = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        #        for aa in range(param.deg_e): projection_i = projection_i + self.hspace.sigma_matrix( self.eg_to_level['e'][aa] , self.eg_to_level['e'][aa] , ii )
        #        for aa in range(param.deg_i): projection_i = projection_i + self.hspace.sigma_matrix( self.eg_to_level['i'][aa] , self.eg_to_level['i'][aa] , ii )
        #        self.H_energies = self.H_energies + param.epsilon_ne2 * projection_i @ ( projection_i - sp.identity(self.hspace.hilbertsize) )
        
        return self.H_energies

    
    
        
        
        
            
            

    
        
    '''
    
    def define_HL_dipolar (self):
        """Saves Hamiltonian and Linbladian part of dipolar interactions."""
        # Incoherent dipole part
        for ii in range(self.Ntotal):
            for a1 in range(param.deg_e):
                for b1 in range(param.deg_g):
                        
                    for jj in range(self.Ntotal):
                        for a2 in range(param.deg_e):
                            for b2 in range(param.deg_g):
                                
                                if self.f_ij(ii,jj,a1,b1,a2,b2)!=0:
                                        
                                    sigma_a1b1_i = self.hspace.sigma_matrix( self.eg_to_level['e'][a1] , self.eg_to_level['g'][b1] , ii )
                                    sigma_a2b2_j = self.hspace.sigma_matrix( self.eg_to_level['e'][a2] , self.eg_to_level['g'][b2] , jj )
                                
                                    #if a1==a1 and b1==b2:
                                    self.ham_eff_dipoles = self.ham_eff_dipoles - 1j * self.f_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i@sigma_a2b2_j.conj().T
                                
                                    self.linSuper_dipoles = self.linSuper_dipoles + 2 * self.f_ij(ii,jj,a1,b1,a2,b2) * sp.kron( sigma_a2b2_j.conj().T , np.transpose(sigma_a1b1_i) , format='csc' )
                                
                                    #self.linSuper = self.linSuper - self.f_ij(ii,jj,a1,b1,a2,b2) * ( sp.kron( sigma_a1b1_i@sigma_a2b2_j.conj().T , one , format='csc' ) \
                                    #                                                                + sp.kron( one , np.transpose(sigma_a1b1_i@sigma_a2b2_j.conj().T) , format='csc' )\
                                    #                                                                - 2 * sp.kron( sigma_a2b2_j.conj().T , np.transpose(sigma_a1b1_i) , format='csc' ) )
        
        # Coherent dipole part
        self.ham_eff_dipoles = self.ham_eff_dipoles + self.define_H_dipole()
                            
    
    def define_H_dipole (self):
        """Returns dipolar Hamiltonian"""
        H_dipole = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        
        for ii in range(self.Ntotal):
            for a1 in range(param.deg_e):
                for b1 in range(param.deg_g):
                        
                    for jj in range(self.Ntotal):
                        for a2 in range(param.deg_e):
                            for b2 in range(param.deg_g):
                                
                                if self.g_ij(ii,jj,a1,b1,a2,b2)!=0:
                                
                                    sigma_a1b1_i = self.hspace.sigma_matrix( self.eg_to_level['e'][a1] , self.eg_to_level['g'][b1] , ii )
                                    sigma_a2b2_j = self.hspace.sigma_matrix( self.eg_to_level['e'][a2] , self.eg_to_level['g'][b2] , jj )
                                
                                    H_dipole = H_dipole - self.g_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i @ sigma_a2b2_j.conj().T
                                
                                    #if ii!=jj:
                                    #    print(a1,b1,a2,b2,self.g_ij(ii,jj,a1,b1,a2,b2))
                                
                                    #if (ii==jj and (a1!=a2 or b1!=b2)): print(a1,a2,b1,b2,self.g_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i @ sigma_a2b2_j.conj().T)
        
        return H_dipole
        
        
    def define_Heff_dipolar (self):
        """Saves effective Hamiltonian of dipolar interactions."""
        self.ham_eff_dipoles = csc_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize) )
        
        # Incoherent dipole part
        for ii in range(self.Ntotal):
            for a1 in range(param.deg_e):
                for b1 in range(param.deg_g):
                        
                    for jj in range(self.Ntotal):
                        for a2 in range(param.deg_e):
                            for b2 in range(param.deg_g):
                                        
                                if self.f_ij(ii,jj,a1,b1,a2,b2)!=0:        
                                #if self.f_ij(ii,ii,a1,b1,a2,b2)!=0:     # cavity      
                                
                                    sigma_a1b1_i = self.hspace.sigma_matrix( self.eg_to_level['e'][a1] , self.eg_to_level['g'][b1] , ii )
                                    sigma_a2b2_j = self.hspace.sigma_matrix( self.eg_to_level['e'][a2] , self.eg_to_level['g'][b2] , jj )
                                
                                    self.ham_eff_dipoles = self.ham_eff_dipoles - 1j * self.f_ij(ii,jj,a1,b1,a2,b2) * sigma_a1b1_i@sigma_a2b2_j.conj().T
                                    #self.ham_eff_dipoles = self.ham_eff_dipoles - 1j * self.f_ij(ii,ii,a1,b1,a2,b2) * sigma_a1b1_i@sigma_a2b2_j.conj().T    # cavity
        
        # Coherent dipole part
        self.ham_eff_dipoles = self.ham_eff_dipoles + self.define_H_dipole()   # comment out w/o cavity
        
        
        # Add extra term to break possible degeneracies of different n_e states
        if param.epsilon_ne!=0:
            for ii in range(self.Ntotal):
                for aa in range(param.deg_e):
                    sigma_aa_i = self.hspace.sigma_matrix( self.eg_to_level['e'][aa] , self.eg_to_level['e'][aa] , ii )
                    #self.H_energies = self.H_energies + ((param.omega0+aa*param.zeeman_e)-param.omegaR) * sigma_aa_i
                    self.ham_eff_dipoles = self.ham_eff_dipoles + param.epsilon_ne * sigma_aa_i
        
        
        # Compute memory
        self.memory_sparse_Heff_dipoles = (self.ham_eff_dipoles.data.nbytes + self.ham_eff_dipoles.indptr.nbytes + self.ham_eff_dipoles.indices.nbytes) / 1024**3        # Number of Bytes  /  Bytes per Gb
        print("Memory for sparse Heff_dipoles: %g Gb."%(self.memory_sparse_Heff_dipoles))
    
    '''
        

####################################################################

##########                ANGULAR MOMENTUM                ##########

####################################################################
    
    def define_Jxyz_collective_sym (self):
        """
        Define COLLECTIVE matrices Jx, Jy, Jz and Jx^2, Jy^2, Jz^2, {Jx,Jy}/2, {Jx,Jz}/2, {Jy,Jz}/2.
        """
        self.define_Jxyz_single_particle()
        
        self.Jc = {}
        self.J2c = {}
        
        # Create arrays
        for alpha in ['x','y','z']:
            self.Jc[alpha] = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize), dtype='complex' )
        for alpha in ['x','y','z']:
            for beta in ['x','y','z']:
                self.J2c[(alpha,beta)] = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize), dtype='complex' )
        
        
        # Fill arrays by going through each entry of single-particle js and j2s
        for aa in range(self.numlevels):
            for bb in range(self.numlevels):
                
                sigma = self.hspace.sigma_ab_sym(aa,bb)
                
                for alpha in ['x','y','z']:
                    self.Jc[alpha] += self.js[alpha][aa,bb] * sigma
                    
                for alpha in ['x','y','z']:
                    for beta in ['x','y','z']:
                        self.J2c[(alpha,beta)] += self.j2s[(alpha,beta)][aa,bb] * sigma
        
        
        # Change sparsity type
        for alpha in ['x','y','z']:
            self.Jc[alpha].tocsc()
        for alpha in ['x','y','z']:
            for beta in ['x','y','z']:
                self.J2c[(alpha,beta)].tocsc()
                
                
            
        
        #print(self.Jc['x'].toarray())
        #print(self.Jc['y'].toarray())
        #print(self.Jc['z'].toarray())
        
        #print(np.sum(np.abs(self.Jc['x']@self.Jc['y'] - self.Jc['y']@self.Jc['x'] - 1j*self.Jc['z'])))
        #print(np.sum(np.abs(self.Jc['y']@self.Jc['z'] - self.Jc['z']@self.Jc['y'] - 1j*self.Jc['x'])))
        #print(np.sum(np.abs(self.Jc['z']@self.Jc['x'] - self.Jc['x']@self.Jc['z'] - 1j*self.Jc['y'])))
        
        #print(np.sum(np.abs(self.J2c[('x','x')]@self.Jc['y'] - self.Jc['y']@self.J2c[('x','x')] - 2*1j*self.J2c[('x','z')])))
        #print(np.sum(np.abs(self.J2c[('y','y')]@self.Jc['z'] - self.Jc['z']@self.J2c[('y','y')] - 2*1j*self.J2c[('y','x')])))
        #print(np.sum(np.abs(self.J2c[('z','z')]@self.Jc['x'] - self.Jc['x']@self.J2c[('z','z')] - 2*1j*self.J2c[('z','y')])))
        
        #print((self.Jc['x']@self.Jc['y'] - self.Jc['y']@self.Jc['x'] - 1j*self.Jc['z']).toarray())
        #print((self.Jc['y']@self.Jc['z'] - self.Jc['z']@self.Jc['y'] - 1j*self.Jc['x']).toarray())
        #print((self.Jc['z']@self.Jc['x'] - self.Jc['x']@self.Jc['z'] - 1j*self.Jc['y']).toarray())
        
        #print((self.J2c[('x','x')]@self.Jc['y'] - self.Jc['y']@self.J2c[('x','x')] - 2*1j*self.J2c[('x','z')]).toarray())
        #print((self.J2c[('y','y')]@self.Jc['z'] - self.Jc['z']@self.J2c[('y','y')] - 2*1j*self.J2c[('y','x')]).toarray())
        #print((self.J2c[('z','z')]@self.Jc['x'] - self.Jc['x']@self.J2c[('z','z')] - 2*1j*self.J2c[('z','y')]).toarray())
    
    
    
    def define_Jxyz_single_particle (self):
        """
        Define matrices Jx, Jy, Jz and Jx^2, Jy^2, Jz^2, {Jx,Jy}/2, {Jx,Jz}/2, {Jy,Jz}/2 for a single particle.
        Coding: 0: x, 1: y, 2: z
        """
        self.js = {}
        self.j2s = {}
        
        jay = (self.numlevels-1)/2
        jay2 = jay * (jay + 1)
        
        # Define Jx
        self.js['x'] = lil_matrix( (self.numlevels,self.numlevels), dtype='complex' )
        for ii in range(0,self.numlevels-1):
            m = ii - jay
            self.js['x'][ii+1,ii] = 1/2 * np.sqrt( jay2 - m*(m+1) )
        for ii in range(1,self.numlevels):
            m = ii - jay
            self.js['x'][ii-1,ii] = 1/2 * np.sqrt( jay2 - m*(m-1) )
            
            
        # Define Jy
        self.js['y'] = lil_matrix( (self.numlevels,self.numlevels), dtype='complex' )
        for ii in range(0,self.numlevels-1):
            m = ii - jay
            self.js['y'][ii+1,ii] = 1/(2*1j) * np.sqrt( jay2 - m*(m+1) )
        for ii in range(1,self.numlevels):
            m = ii - jay
            self.js['y'][ii-1,ii] = -1/(2*1j) * np.sqrt( jay2 - m*(m-1) )
            
            
        # Define Jz
        self.js['z'] = lil_matrix( (self.numlevels,self.numlevels), dtype='complex' )
        for ii in range(0,self.numlevels):
            m = ii - jay
            self.js['z'][ii,ii] = m
            
            
        # Define 1
        
        
        
        #print(self.js['x'].toarray())
        #print(self.js['y'].toarray())
        #print(self.js['z'].toarray())
        
        
        # Define quadratic operators
        self.j2s[('x','x')] = self.js['x'] @ self.js['x']
        self.j2s[('y','y')] = self.js['y'] @ self.js['y']
        self.j2s[('z','z')] = self.js['z'] @ self.js['z']
        
        self.j2s[('x','y')] = 1/2 * ( self.js['x'] @ self.js['y'] + self.js['y'] @ self.js['x'] )
        self.j2s[('x','z')] = 1/2 * ( self.js['x'] @ self.js['z'] + self.js['z'] @ self.js['x'] )
        self.j2s[('y','z')] = 1/2 * ( self.js['y'] @ self.js['z'] + self.js['z'] @ self.js['y'] )
        
        self.j2s[('y','x')] = self.j2s[('x','y')]
        self.j2s[('z','x')] = self.j2s[('x','z')]
        self.j2s[('z','y')] = self.j2s[('y','z')]
        
        #print(self.j2s[('x','y')].toarray())
        #print(self.j2s[('x','z')].toarray())
        #print(self.j2s[('y','z')].toarray())
        
        
        # Apply random unitary
        #u = unitary_group.rvs(self.numlevels)
        u = np.eye(self.numlevels)
        
        for alpha in ['x','y','z']:
            self.js[alpha] = u @ self.js[alpha] @ u.conj().T
        for alpha in ['x','y','z']:
            for beta in ['x','y','z']:
                self.j2s[(alpha,beta)] = u @ self.j2s[(alpha,beta)] @ u.conj().T
        
        
        
    
    
    
    def define_gellmann_collective_sym (self):
        """
        Permutationally symmetric hilbert space.
        Creates dictionary of gm_i and (gm_i gm_j + gm_j gm_i)/2 operators for each Gell-Mann matrix of local hilbert space.
        """
        print('Computing Gell-Mann')
        
        self.define_gellmann_single_particle()
        
        self.GMc = {}
        self.GM2c = {}
        
        # Create arrays
        for alpha in range(len(self.gms)):
            self.GMc[alpha] = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize), dtype='complex' )
        for alpha in range(len(self.gms)):
            for beta in range(len(self.gms)):
                self.GM2c[(alpha,beta)] = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize), dtype='complex' )
        
        
        # Fill arrays by going through each entry of single-particle gms and gm2s
        for aa in range(self.numlevels):
            for bb in range(self.numlevels):
                
                sigma = self.hspace.sigma_ab_sym(aa,bb)
                
                for alpha in range(len(self.gms)):
                    if self.gms[alpha][aa,bb] != 0:
                        self.GMc[alpha] += self.gms[alpha][aa,bb] * sigma
                    
                for alpha in range(len(self.gms)):
                    for beta in range(len(self.gms)):
                        if self.gm2s[(alpha,beta)][aa,bb] != 0:
                            self.GM2c[(alpha,beta)] += self.gm2s[(alpha,beta)][aa,bb] * sigma
        
        print('Done')
        
        # Change sparsity type
        for alpha in range(len(self.gms)):
            self.GMc[alpha].tocsc()
        for alpha in range(len(self.gms)):
            for beta in range(len(self.gms)):
                self.GM2c[(alpha,beta)].tocsc()
    
    
    
    def define_gellmann_single_particle (self):
        """
        Define Gell-Mann matrices gm_i and gm_i^2, {gm_i,gm_j}/2 for a single particle.
        Use normalized matrices. For diagonal use traceful matrices.
        's' stands for 'single'
        """
        self.gms = {}
        self.gm2s = {}
        
        # Define gell-mann operators
        temp = []
        
        # Z
        for aa in range(self.numlevels):
            sz = np.zeros((self.numlevels,self.numlevels),dtype=complex)
                # lil_matrix( (self.numlevels,self.numlevels), dtype='complex' )
            sz[aa,aa] = 1
            temp.append(sz)
    
        # X, Y
        for aa in range(self.numlevels):
            for bb in range(aa+1,self.numlevels):
        
                sx = np.zeros((self.numlevels,self.numlevels),dtype=complex)
                sy = np.zeros((self.numlevels,self.numlevels),dtype=complex)
        
                sx[aa,bb] = 1/sqrt(2)
                sx[bb,aa] = 1/sqrt(2)
        
                sy[aa,bb] = 1j/sqrt(2)
                sy[bb,aa] = -1j/sqrt(2)
        
                temp.append(sx)
                temp.append(sy)
        
        # Trafo into dictionary
        for ii in range(len(temp)):
            self.gms[ii] = temp[ii]
        
        # Define quadratic operators
        for ii in range(len(self.gms)):
            for jj in range(len(self.gms)):
                
                self.gm2s[(ii,jj)] = 1/2 * ( self.gms[ii] @ self.gms[jj] + self.gms[jj] @ self.gms[ii] )
    
    
    
    
    
    
    def define_complex_gellmann_collective_sym (self):
        """
        Permutationally symmetric hilbert space.
        Creates dictionary of gm_i and (gm_i gm_j + gm_j gm_i)/2 operators for each complex Gell-Mann matrix of local hilbert space.
        """
        print('Computing complex (non-hermitian) Gell-Mann')
        
        self.define_complex_gellmann_single_particle()
        
        self.cxGMc = {}
        self.cxGM2c = {}
        
        # Create arrays
        for alpha in range(len(self.cxgms)):
            self.cxGMc[alpha] = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize), dtype='complex' )
        for alpha in range(len(self.cxgms)):
            for beta in range(len(self.cxgms)):
                self.cxGM2c[(alpha,beta)] = lil_matrix( (self.hspace.hilbertsize,self.hspace.hilbertsize), dtype='complex' )
        
        
        # Fill arrays by going through each entry of single-particle cxgms and cxgm2s
        for aa in range(self.numlevels):
            for bb in range(self.numlevels):
                
                sigma = self.hspace.sigma_ab_sym(aa,bb)
                
                for alpha in range(len(self.cxgms)):
                    if self.cxgms[alpha][aa,bb] != 0:
                        self.cxGMc[alpha] += self.cxgms[alpha][aa,bb] * sigma
                    
                for alpha in range(len(self.cxgms)):
                    for beta in range(len(self.cxgms)):
                        if self.cxgm2s[(alpha,beta)][aa,bb] != 0:
                            self.cxGM2c[(alpha,beta)] += self.cxgm2s[(alpha,beta)][aa,bb] * sigma
        
        print('Done')
        
        # Change sparsity type
        for alpha in range(len(self.cxgms)):
            self.cxGMc[alpha].tocsc()
        for alpha in range(len(self.cxgms)):
            for beta in range(len(self.cxgms)):
                self.cxGM2c[(alpha,beta)].tocsc()
    
    
    
    def define_complex_gellmann_single_particle (self):
        """
        Define complex Gell-Mann matrices gm_i and gm_i^2, {gm_i,gm_j}/2 for a single particle.
        Use normalized matrices. For diagonal use traceful matrices.
        's' stands for 'single'
        """
        self.cxgms = {}
        self.cxgm2s = {}
        
        # Define gell-mann operators
        temp = []
    
        # "Complex" non-hermitian Gell-Mann matrices
        for aa in range(self.numlevels):
            for bb in range(self.numlevels):
        
                gell = np.zeros((self.numlevels,self.numlevels),dtype=complex)
        
                gell[aa,bb] = 1
        
                temp.append(gell)
        
        # Trafo into dictionary
        for ii in range(len(temp)):
            self.cxgms[ii] = temp[ii]
        
        # Define quadratic operators
        for ii in range(len(self.cxgms)):
            for jj in range(len(self.cxgms)):
                
                self.cxgm2s[(ii,jj)] = 1/2 * ( self.cxgms[ii] @ (self.cxgms[jj]).conj().T + (self.cxgms[jj]).conj().T @ self.cxgms[ii] )
    
    
    
    
    '''
    
    ### Old definition of Gell-Mann
        
    def define_gell_mann_sym (self):
        """
        Permutationally symmetric hilbert space.
        Creates list of lambda_i and (lambda_i lambda_j + lambda_j lambda_i)/2 operators for each Gell-Mann matrix of local hilbert space.
        """
        
        # Compute COLLECTIVE Gell-Mann matrices
        
        temp1 = []
        temp2 = []
        
        # Mean
        # Z
        for aa in range(self.hspace.localhilbertsize):
            sz = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            # Construct sz = |aa><aa|
            for state in self.hspace.state_to_index:
                sz += state[aa] * self.hspace.transitionOp_sym(state,state)
            sz = sz / self.Ntotal
            temp1.append( sz )
            
        # X, Y
        for aa in range(self.hspace.localhilbertsize):
            for bb in range(aa+1,self.hspace.localhilbertsize):
                
                sx = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
                sy = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize) , dtype='complex')
                
                # Construct sx = |bb><aa| + |aa><bb| and sy = -i*|bb><aa| + i*|aa><bb|  (and later normalize by sqrt(2))
                changestate = np.zeros(self.numlevels, dtype='int')
                changestate[aa] = -1
                changestate[bb] = 1
                
                for state in self.hspace.state_to_index:
                    # |bb><aa|
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate + changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        sx[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += sqrt( rstate[aa] * (rstate[bb]+1) )
                        sy[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += -1j*sqrt( rstate[aa] * (rstate[bb]+1) )
                        
                    # |aa><bb|
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate - changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        sx[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += sqrt( rstate[bb] * (rstate[aa]+1) )
                        sy[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += 1j*sqrt( rstate[bb] * (rstate[aa]+1) )
                
                # Normalize
                sx = sx / self.Ntotal / sqrt(2)
                sy = sy / self.Ntotal / sqrt(2)
                
                temp1.append( sx )
                temp1.append( sy )
                
                
        # Variances
        for ii in range(len(temp1)):
            for jj in range(ii,len(temp1)):
                
                temp2.append( ( temp1[ii]@temp1[jj] + temp1[jj]@temp1[ii] )/2 )
        
        # Adding everything to output
        #if 'gell_mann' in param.which_observables:
        #    self.output_ops.extend( self.GMc )
        #    self.output_ops.extend( self.GM2c )        
        
        # Save Gell-Mann matrices
        self.GMc = temp1
        self.GM2c = temp2
        
        # Compute SINGLE-PARTICLE Gell-Mann matrices, in the same order as operators self.GMc
        self.gms = []
        # Z
        for aa in range(self.numlevels):
            sz = np.zeros((self.numlevels,self.numlevels),dtype=complex)
            sz[aa,aa] = 1
            self.gms.append(sz)
    
        # X, Y
        for aa in range(self.numlevels):
            for bb in range(aa+1,self.numlevels):
        
                sx = np.zeros((self.numlevels,self.numlevels),dtype=complex)
                sy = np.zeros((self.numlevels,self.numlevels),dtype=complex)
        
                sx[aa,bb] = 1/sqrt(2)
                sx[bb,aa] = 1/sqrt(2)
        
                sy[aa,bb] = 1j/sqrt(2)
                sy[bb,aa] = -1j/sqrt(2)
        
                self.gms.append(sx)
                self.gms.append(sy)


    '''



####################################################################

############                EIGENSTATES                #############

####################################################################



    def truncate(self,number, digits) -> float:
        stepper = pow(10.0, digits)
        return math.trunc(stepper * number) / stepper
        
        
    def save_only_H_cavity (self):
        """
        Saves only H_cavity for computation of eigenstates.
        """
        if self.Pi_Sigma_defined==False: self.define_Pi_Sigma()
        
        self.define_H_cavity()
        
        if param.hilberttype=='full':
            self.define_H_energies()
        if param.hilberttype=='sym':
            self.define_H_energies_sym()
        
        self.ham_eff_total = self.ham_eff_cavity + self.H_energies
        
        
    def compute_eigenstates (self):
        """
        Compute all eigenvalues and eigenvectors of cavity effective Hamiltonian
        """
        if self.memory_full_Hamiltonian<param.max_memory:
            self.evalues,self.estates = eig(self.ham_eff_total.toarray())
        elif self.memory_sparse_Hamiltonian<param.max_memory:
            print("WARNING/compute_eigenstates: Memory of full Hamiltonian too large. Using eigs, list of eigenstates incomplete.")
            self.evalues,self.estates = eigs(self.ham_eff_total,k=len(self.hspace.hilbertsize)-2,which='SR')
        else: print("ERROR/compute_eigenstates: Memory of sparse Hamiltonian too large. Eigenstates not computed.")
        
        self.compute_excitations_of_eigenstates()
        
        """
        for ii in range(len(self.evalues)):
            if self.excitations_estates[ii]>0:
                if abs(self.evalues[ii].imag)<0.0000001:                    
                    self.darkstate = self.estates[:,ii]
        """         
        #print("\n *************")
        #print("This is the dark state that will be saved:\n")
        #print(self.darkstate)
        #print("\n *************")
        
    
    
    def compute_excitations_of_eigenstates (self):
        """
        SYMMETRIC manifold
        Computes to which excitation manifold each eigenstate belongs.
        For H_cavity each eigenstate is a superposition of states with the same number of excitations.
        If eigenstates mix different manifolds, print error.
        
        Also saves how many symmetric states are involved in each eigenstate.
        """
        self.excitations_estates = np.full((len(self.evalues)),-1,dtype=int)
        self.nstatesinvolved_estates = np.full((len(self.evalues)),-1,dtype=int)
        
        for ii in range(len(self.evalues)):
            # Get eigenstate, set all small entries to 0, and for the remaining retain only absolute value
            trunc_estate = np.array( [ self.truncate(abs(self.estates[jj,ii]),param.digits_trunc) for jj in range(len(self.estates[:,ii])) ] )
            
            # Save number of states contributing to eigenstate
            involved_states = []
            weight_involved_states = []
            for jj in range(len(trunc_estate)):
                if trunc_estate[jj]!=0:
                    involved_states.append(jj)
                    weight_involved_states.append(trunc_estate[jj]**2)
            self.nstatesinvolved_estates[ii] = len(involved_states)
    
            # Compute excitations of involved states, check all excitations are the same, and save value of excitations.
            if len(involved_states)>0:
                excit_involved_states = [ self.hspace.index_to_number_excitations[involved_states[jj]] for jj in range(len(involved_states)) ]
                
                # Check if all states involved have same number of excitations and save value (or mean value)
                if excit_involved_states.count(excit_involved_states[0]) == len(excit_involved_states):
                    self.excitations_estates[ii] = excit_involved_states[0]
                else:
                    meanexcit = sum( [ excit_involved_states[jj]*weight_involved_states[jj] for jj in range(len(involved_states)) ] )
                    self.excitations_estates[ii] = round( meanexcit )
                    print(meanexcit)
                    
                    #indexmax = max(range(len(weight_involved_states)), key=weight_involved_states.__getitem__)
                    
                    print("WARNING/compute_excitations_of_eigenstates: Eigenstates mix different excitation manifolds.")
                
            else: print("ERROR/compute_excitations_of_eigenstates: Entries of eigenstate %i seem to be zero."%(ii))
        
    
    
    def compute_Renyi_darkstates_fixed_k_m (self):
        """
        SYMMETRIC manifold
        Project Hamiltonian onto subspaces of fixed number of excitations k and fixed magnetic number m
        This ONLY makes sense in the PARALLEL BASIS, where m_parallel is conserved.
        Compute eigenstates
        Compute purity and Gell-Mann expectation values.
        Compute Renyi of dark states.
        """
        
        out = []
        
        for kk in range(self.Ntotal+1):
            
            min_mm = kk*self.Mes[0] + (self.Ntotal-kk)*self.Mgs[0]
            max_mm = kk*self.Mes[-1] + (self.Ntotal-kk)*self.Mgs[-1]
            
            steps = max_mm - min_mm
            mms = [ min_mm+mm for mm in range(steps+1) ]
            
            ms_array = np.array(self.Mgs+self.Mes)
            
            for mm in mms:
                
                indices_kk_mm = []
            
                # Save indices of states with kk excitations and mm magnetic number
                for index in self.hspace.index_to_state:    # Run over all states
            
                    state = self.hspace.index_to_state[index]
                    state_array = np.array(state)
                
                    if sum(state[param.deg_g:])==kk:
                        
                        if np.sum(state_array*ms_array)==mm:
                        
                            indices_kk_mm.append(index)
        
                # Project Hamiltonian
                projected_Heff = self.ham_eff_total[np.ix_(indices_kk_mm,indices_kk_mm)]
                
                # Calculate eigenvalues and eigenstates
                km_evalues,km_estates = eig(projected_Heff.toarray())
                
                # Compute Renyi of dark eigenstates
                for ii in range(len(km_evalues)):
                    
                    if abs(km_evalues[ii].imag)<0.0000000001:
                        
                        darkstate = km_estates[:,ii].reshape((-1,1))
                        darkstate = darkstate / normLA(darkstate)
                        
                        # Project Gell-Mann operators
                        proj_gell_mann = []
                        for nn in range(self.numlevels**2):
                            proj_gell_mann.append( self.GMc[nn][np.ix_(indices_kk_mm,indices_kk_mm)] )
    
                        # Compute Gell-Mann means
                        GM_mean = [  ( np.squeeze( darkstate.conj().T @ proj_gell_mann[nn] @ darkstate ) ).real  for  nn in range(self.numlevels**2)  ]
    
                        # Compute reduced 1-particle density matrix
                        rho1 = np.zeros((self.numlevels,self.numlevels),dtype=complex)
    
                        for jj in range(self.numlevels**2):
        
                            lambda_j = GM_mean[jj]
                            rho1 += lambda_j * self.gms[jj]

                        # Compute single-particle Renyi entropy
                        renyi_1_dark = -np.log2( ((rho1@rho1).diagonal().sum()).real )
                        
                        print([kk, mm, -2*km_evalues[ii].imag, renyi_1_dark])
                        out.append([kk, mm, -2*km_evalues[ii].imag, renyi_1_dark])
            
        return out
        
        
####################################################################

############                DYNAMICS                ###############

####################################################################
       
    def linblad_equation(self,t,psi):
        return self.linSuper @ psi
       
    
    def compute_evolution_op (self):
            
        #self.evolU = expm(self.linSuper.toarray()*param.dt)
        self.evolU = sp_expm(self.linSuper*param.dt)
    
    def evolve_rho_onestep (self):
        
        if param.solver == 'exp':
            self.rhoV = self.evolU @ self.rhoV
            self.rho = self.rhoV.reshape(self.hspace.hilbertsize,self.hspace.hilbertsize)
        
        if param.solver == 'ode':
            self.solver.integrate(self.solver.t+param.dt)
            self.rho = self.solver.y.reshape(self.hspace.hilbertsize,self.hspace.hilbertsize)
            self.rhoV = self.rho.flatten('C').reshape(self.hspace.hilbertsize**2,1)


    def set_solver (self,time):
        """
        Choose solver for ODE and set initial conditions.
        """
        self.solver = complex_ode(self.linblad_equation).set_integrator('dopri5',atol=param.atol,rtol=param.rtol)
        self.solver.set_initial_value(self.rhoV.reshape((len(self.rhoV))), time)
        
        print('atol:', self.solver._integrator.atol)
        print('rtol:', self.solver._integrator.rtol)
        
        
        



####################################################################

#############                OUTPUT                ################

####################################################################
    
    
    def create_output_ops_occs (self):
        """
        Creates list of occupation operators whose expectation value will be outputed.
        """
        self.output_ops = []
        self.complex_output_ops = []
        
        if param.hilberttype=='full':
            if 'populations' in param.which_observables:
                self.create_output_ops_occs_full()
            if 'pop_variances' in param.which_observables:
                print("\nWARNING: pop_variances type of which_observables not valid for full hilbertspace (not implemented).\n")
            if 'xyz' in param.which_observables:
                print("\nWARNING: xyz type of which_observables not valid for full hilbertspace (not implemented).\n")
                
        if param.hilberttype=='sym':
            if 'populations' in param.which_observables:
                self.create_output_ops_occs_sym()
            if 'pop_variances' in param.which_observables:
                self.create_population_variances_output_sym()
            if 'xyz' in param.which_observables:
                self.create_XYZ_output_sym()
            
            
        if 'photons' in param.which_observables:
            self.output_ops.append( self.Pi_plus @ self.Pi_minus )
            self.output_ops.append( self.Sigma_plus @ self.Sigma_minus )
            self.output_ops.append( self.Pi_plus @ self.Sigma_minus + self.Sigma_plus @ self.Pi_minus )
            self.output_ops.append( 1j * self.Pi_plus @ self.Sigma_minus - 1j * self.Sigma_plus @ self.Pi_minus )
            
            
        if param.hilberttype=='full':
            if 'xyz_variances' in param.which_observables:
                print("\nWARNING: xyz_variances type of which_observables not valid for full hilbertspace (not implemented).\n")
            if 'gell_mann' in param.which_observables:
                print("\nWARNING: gell_mann type of which_observables not valid for full hilbertspace (not implemented).\n")
                
        if param.hilberttype=='sym':
            if 'xyz_variances' in param.which_observables:
                self.create_XYZ_variances_output_sym()
            if 'gell_mann' in param.which_observables:
                #self.output_ops.extend( list(self.GMc.values()) )
                #self.output_ops.extend( list(self.GM2c.values()) )
                    
                self.complex_output_ops.extend( list(self.cxGMc.values()) )
                for ii in range(len(self.cxGMc)):
                    for jj in range(len(self.cxGMc)):
                        
                        self.complex_output_ops.append( 1/2 * ( self.cxGMc[ii] @ (self.cxGMc[jj]).conj().T + (self.cxGMc[jj]).conj().T @ self.cxGMc[ii] ) )
                        
                #self.complex_output_ops.extend( list(self.cxGM2c.values()) )
                
                
                
        
    def create_output_ops_occs_full (self):
        """
        Full hilbert space:
        Creates list of occupation operators whose expectation value will be outputed.
        """
        for nn in range(self.hspace.localhilbertsize):
            observable = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            for ii in range(self.Ntotal):
                projector_i = self.hspace.transitionOp_onsite(nn,nn,ii)
                observable = observable + projector_i
            self.output_ops.append( observable / self.Ntotal )
            
            
    def create_output_ops_occs_sym (self):
        """
        Permutationally symmetric hilbert space:
        Creates list of occupation operators whose expectation value will be outputed.
        """
        for nn in range(self.hspace.localhilbertsize):
            observable = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            
            # Construct projector |nn><nn|
            for state in self.hspace.state_to_index:
                observable += state[nn] * self.hspace.transitionOp_sym(state,state)
                
            self.output_ops.append( observable / self.Ntotal )
            
            
            
        
    def create_XYZ_output_sym (self):
        """
        Permutationally symmetric hilbert space.
        Creates list of pauli_x, pauli_y and pauli_z operators for each pair of states in local hilbert space.
        """
        
        for aa in range(self.hspace.localhilbertsize):
            for bb in range(aa+1,self.hspace.localhilbertsize):
                
                sx = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
                sy = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize) , dtype='complex')
                sz = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
                
                # Construct sz = |bb><bb| - |aa><aa|
                for state in self.hspace.state_to_index:
                    sz += ( state[bb] - state[aa] ) * self.hspace.transitionOp_sym(state,state)
                    
                # Construct sx = |bb><aa| + |aa><bb| and sy = -i*|bb><aa| + i*|aa><bb|
                changestate = np.zeros(self.numlevels, dtype='int')
                changestate[aa] = -1
                changestate[bb] = 1
                
                for state in self.hspace.state_to_index:
                    # |bb><aa|
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate + changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        sx[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += sqrt( rstate[aa] * (rstate[bb]+1) )
                        sy[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += -1j*sqrt( rstate[aa] * (rstate[bb]+1) )
                        
                    # |aa><bb|
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate - changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        sx[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += sqrt( rstate[bb] * (rstate[aa]+1) )
                        sy[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += 1j*sqrt( rstate[bb] * (rstate[aa]+1) )
                
                # Normalize
                sx = sx / self.Ntotal
                sy = sy / self.Ntotal
                sz = sz / self.Ntotal
                
                # Add to output
                self.output_ops.append( sx )
                self.output_ops.append( sy )
                self.output_ops.append( sz )
                
                
                
    def create_population_variances_output_sym (self):
        """
        Permutationally symmetric hilbert space.
        Creates list of na^2,... , na*nb, ... population operators for each pair of states in local hilbert space.
        """
        
        # Create population operators and add nocc^2
        nocc = [ csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize)) for aa in range(self.hspace.localhilbertsize) ]
        
        for aa in range(self.hspace.localhilbertsize):
            
            nocc[aa] = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            
            # Construct na = |aa><aa|
            for state in self.hspace.state_to_index:
                nocc[aa] +=  state[aa] * self.hspace.transitionOp_sym(state,state)
                
            # Normalize
            nocc[aa] = nocc[aa] / self.Ntotal
            
            # Add to output
            self.output_ops.append( nocc[aa]@nocc[aa] )
        
        
        # Add nocc[aa]*nocc[bb]
        for aa in range(self.hspace.localhilbertsize):
            for bb in range(aa+1,self.hspace.localhilbertsize):
                
                # Add to output
                self.output_ops.append( (nocc[aa]@nocc[bb]+nocc[bb]@nocc[aa])/2 )
                
                
                
    def create_XYZ_variances_output_sym (self):
        """
        Permutationally symmetric hilbert space.
        Creates list of sx^2, sy^2, sz^2, (sxsy+sysx)/2, (sxsz+szsx)/2, (sysz+szsy)/2 operators for each pair of states in local hilbert space.
        """
        
        for aa in range(self.hspace.localhilbertsize):
            for bb in range(aa+1,self.hspace.localhilbertsize):
                
                sx = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
                sy = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize) , dtype='complex')
                sz = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
                
                # Construct sz = |bb><bb| - |aa><aa|
                for state in self.hspace.state_to_index:
                    sz += ( state[bb] - state[aa] ) * self.hspace.transitionOp_sym(state,state)
                    
                # Construct sx = |bb><aa| + |aa><bb| and sy = -i*|bb><aa| + i*|aa><bb|
                changestate = np.zeros(self.numlevels, dtype='int')
                changestate[aa] = -1
                changestate[bb] = 1
                
                for state in self.hspace.state_to_index:
                    # |bb><aa|
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate + changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        sx[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += sqrt( rstate[aa] * (rstate[bb]+1) )
                        sy[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += -1j*sqrt( rstate[aa] * (rstate[bb]+1) )
                        
                    # |aa><bb|
                    rstate = np.array( state , dtype='int' )
                    lstate = rstate - changestate
                    if tuple(lstate) in self.hspace.state_to_index:
                        sx[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += sqrt( rstate[bb] * (rstate[aa]+1) )
                        sy[ self.hspace.state_to_index[tuple(lstate)] , self.hspace.state_to_index[tuple(rstate)] ] += 1j*sqrt( rstate[bb] * (rstate[aa]+1) )
                
                # Normalize
                sx = sx / self.Ntotal
                sy = sy / self.Ntotal
                sz = sz / self.Ntotal
                
                # Add to output
                self.output_ops.append( sx@sx )
                self.output_ops.append( sy@sy )
                self.output_ops.append( sz@sz )
                self.output_ops.append( (sx@sy+sy@sx)/2 )
                self.output_ops.append( (sx@sz+sz@sx)/2 )
                self.output_ops.append( (sy@sz+sz@sy)/2 )
        
                
                


    def read_occs (self):
        """
        Outputs occupation of each Hilbert state, summed over all atoms.
        """
        out = []
        # Dark general F
        #out.append( (self.darkstate.conj().T @ self.rho @ self.darkstate).real )
        #if param.deg_e==2 and param.deg_g==2:
        #    out.append( (((self.rho[2,2]+self.rho[2,3]) + (self.rho[3,2]+self.rho[3,3]))/2).real )
        
        # Output of Hermitian operators
        for ii in range(len(self.output_ops)):
            
            out.append( ( (self.output_ops[ii] @ self.rho).diagonal().sum() ).real )
            
        # Output of non-Hermitian operators
        for ii in range(len(self.complex_output_ops)):
            
            expV = (self.complex_output_ops[ii] @ self.rho).diagonal().sum()
            
            out.append( expV.real )
            out.append( expV.imag )
            
        
        # Compute quantum Fisher information
        if param.hilberttype=='sym':
            if 'fisher' in param.which_observables:
                if self.memory_full_Hamiltonian<param.max_memory:
                    
                    #evalues,estates = eig(self.rho)
                    print("Diagonalizing rho")
                    evalues,estates = eigh(self.rho)
                    print("done")
                    
                    mixed_cov = np.zeros((len(self.GMc),len(self.GMc)), dtype=complex)
                    
                    print("Evaluating sum over eigenstates")
                    
                    #max_eigv = np.amax(evalues)
                    #epsilon = 0.01
                    #cutoff = max_eigv*epsilon
                    
                    for ii in range(len(self.GMc)):
                        for jj in range(ii,len(self.GMc)):
                    
                            for kk in range(len(evalues)):
                                for qq in range(len(evalues)):
                                    
                                    # Doesn't work. Maybe there is an error
                                    #if evalues[kk]>=cutoff and evalues[qq]<cutoff:
                                    #    mixed_cov[ii,jj] += 2 * (estates[:,qq].conj().T @ self.GMc[jj] @ self.rho @ self.GMc[ii] @ estates[:,qq])
                                    #    
                                    #elif evalues[kk]<cutoff and evalues[qq]>=cutoff:
                                    #    mixed_cov[ii,jj] += 2 * (estates[:,kk].conj().T @ self.GMc[ii] @ self.rho @ self.GMc[jj] @ estates[:,kk])
                                    #                        
                                    #elif evalues[kk]>=cutoff and evalues[qq]>=cutoff:
                                    #    mixed_cov[ii,jj] += 2 * (evalues[kk]-evalues[qq])**2 / (evalues[kk]+evalues[qq]) \
                                    #                        * (estates[:,kk].conj().T @ self.GMc[ii] @ estates[:,qq]) \
                                    #                        * (estates[:,qq].conj().T @ self.GMc[jj] @ estates[:,kk])
                                    
                                    if evalues[kk]+evalues[qq]>0.000001:    
                                        mixed_cov[ii,jj] += 2 * (evalues[kk]-evalues[qq])**2 / (evalues[kk]+evalues[qq]) \
                                                            * (estates[:,kk].conj().T @ self.GMc[ii] @ estates[:,qq]) \
                                                            * (estates[:,qq].conj().T @ self.GMc[jj] @ estates[:,kk])
                                    
                    for ii in range(len(self.GMc)):
                        for jj in range(ii):
                            mixed_cov[ii,jj] = np.conj(mixed_cov[jj,ii])
                            
                    print("done")
                      
                    evals_cov = eigvalsh(mixed_cov)
                    out.append( np.amax(evals_cov) )
                            
                    
                else:
                    print("\nERROR/read_occs: Fisher information can not be computed, memory of full Hamiltonian is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
                    sys.exit()
        
        
        return out
        
    
    def read_groundstate_distribution (self):
        """
        Output diagonal of density matrix as function of population of state (only states with no excitations)
        """
        
        out = []
        
        for index in self.hspace.index_to_state:    # Run over all states
            
            state = self.hspace.index_to_state[index]
            value = (self.rho[index,index]).real
            
            if sum(state[param.deg_g:])==0:  # If no excitations
            
                out.append( [ state[aa] for aa in range(param.deg_g) ] + [ value ] )
        
        return out
        
        
    def read_fullstate_distribution (self):
        """
        Output diagonal of density matrix as function of population of state (all states)
        """
        
        out = []
        
        for index in self.hspace.index_to_state:    # Run over all states
            
            state = self.hspace.index_to_state[index]
            value = (self.rho[index,index]).real
            
            out.append( [ state[aa] for aa in range(param.deg_g+param.deg_e) ] + [ value ] )
        
        return out
        
    
    
    ###########     PURITY and RENYI        ###############
    
    def compute_purity_and_Renyi (self):
        """
        Compute purity and Gell-Mann expectation values.
        Compute Renyi.
        """
        out = []
        
        trace = (np.trace(self.rho)).real
        purity = (np.trace(self.rho@self.rho)).real
        
        # Compute Gell-Mann means
        GM_mean = [  ( (self.GMc[ii] @ self.rho).diagonal().sum() ).real  for  ii in range(self.numlevels**2)  ]
        
        # Compute reduced 1-particle density matrix
        rho1 = np.zeros((self.numlevels,self.numlevels),dtype=complex)
        
        for jj in range(self.numlevels**2):
            
            lambda_j = GM_mean[jj]
            rho1 += lambda_j * self.gms[jj]

        # Compute single-particle Renyi entropy
        purity_1 = (np.trace(rho1@rho1)).real
        renyi_1 = -np.log2( purity_1 ) #((rho1@rho1).diagonal().sum()).real 
        
        # Compute overlap with initial state
        overlap = (np.trace( self.rho @ self.rho_t0 )).real
        
        out = [trace, purity, overlap, renyi_1]
        
        
        #print([trace, purity, purity_1])
        
        return out
        
        #print([trace,purity,renyi_1])
        
        
    def save_initial_rho (self):
        """
        Save rho(t=0) to compute overlap with final state
        """
        self.rho_t0 = self.rho
    
    
    def compute_purity_and_Renyi_of_fixed_excitation (self):
        """
        Compute the density matrix projected onto a fixed number of excitations k.
        Normalize it.
        Compute purity and Gell-Mann expectation values.
        Compute Renyi.
        """
        
        out = []
        
        for kk in range(self.Ntotal+1):
            
            indices_kk = []
            #proj_kk = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
            
            # Save indices of states with kk excitations
            for index in self.hspace.index_to_state:    # Run over all states
            
                state = self.hspace.index_to_state[index]
                
                if sum(state[param.deg_g:])==kk:   indices_kk.append(index)
        
            # Project rho
            projected_rho = self.rho[np.ix_(indices_kk,indices_kk)]
            
            # Normalize
            trace_prho = (np.trace(projected_rho)).real
            projected_rho = projected_rho / trace_prho
            
            # Purity
            purity_prho = (np.trace(projected_rho@projected_rho)).real
            
            
            
            
            # Project Gell-Mann operators
            proj_gell_mann = []
            for ii in range(self.numlevels**2):
                proj_gell_mann.append( self.GMc[ii][np.ix_(indices_kk,indices_kk)] )
        
            # Compute Gell-Mann means
            GM_mean = [  ( (proj_gell_mann[ii] @ projected_rho).diagonal().sum() ).real  for  ii in range(self.numlevels**2)  ]
        
            # Compute reduced 1-particle density matrix
            rho1 = np.zeros((self.numlevels,self.numlevels),dtype=complex)
        
            for jj in range(self.numlevels**2):
            
                lambda_j = GM_mean[jj]
                rho1 += lambda_j * self.gms[jj]

            # Compute single-particle Renyi entropy
            renyi_1_prho = -np.log2( ((rho1@rho1).diagonal().sum()).real )
            
            
            
            out.append([kk, trace_prho, purity_prho, renyi_1_prho])
            
        return out
            
    
    def compute_purity_and_Renyi_of_fixed_k_m (self):
        """
        Compute the density matrix projected onto a fixed number of excitations k and fixed magnetic number m
        This ONLY makes sense in the PARALLEL BASIS, where m_parallel is conserved.
        Normalize it.
        Compute purity and Gell-Mann expectation values.
        Compute Renyi.
        """
        
        out = []
        
        for kk in range(self.Ntotal+1):
            
            min_mm = kk*self.Mes[0] + (self.Ntotal-kk)*self.Mgs[0]
            max_mm = kk*self.Mes[-1] + (self.Ntotal-kk)*self.Mgs[-1]
            
            steps = max_mm - min_mm
            mms = [ min_mm+mm for mm in range(steps+1) ]
            
            ms_array = np.array(self.Mgs+self.Mes)
            
            for mm in mms:
                
                indices_kk_mm = []
            
                # Save indices of states with kk excitations and mm magnetic number
                for index in self.hspace.index_to_state:    # Run over all states
            
                    state = self.hspace.index_to_state[index]
                    state_array = np.array(state)
                
                    if sum(state[param.deg_g:])==kk:
                        
                        if np.sum(state_array*ms_array)==mm:
                        
                            indices_kk_mm.append(index)
        
                # Project rho
                projected_rho = self.rho[np.ix_(indices_kk_mm,indices_kk_mm)]
            
                # Trace
                trace_prho = (np.trace(projected_rho)).real
                
                if trace_prho>0:
                    #Normalize
                    projected_rho = projected_rho / trace_prho
            
                    # Purity
                    purity_prho = (np.trace(projected_rho@projected_rho)).real
                    
                    # Project Gell-Mann operators
                    proj_gell_mann = []
                    for ii in range(self.numlevels**2):
                        proj_gell_mann.append( self.GMc[ii][np.ix_(indices_kk_mm,indices_kk_mm)] )
        
                    # Compute Gell-Mann means
                    GM_mean = [  ( (proj_gell_mann[ii] @ projected_rho).diagonal().sum() ).real  for  ii in range(self.numlevels**2)  ]
        
                    # Compute reduced 1-particle density matrix
                    rho1 = np.zeros((self.numlevels,self.numlevels),dtype=complex)
        
                    for jj in range(self.numlevels**2):
            
                        lambda_j = GM_mean[jj]
                        rho1 += lambda_j * self.gms[jj]

                    # Compute single-particle Renyi entropy
                    renyi_1_prho = -np.log2( ((rho1@rho1).diagonal().sum()).real )
                    
                else:
                    purity_prho = 1
                    renyi_1_prho = 0
                
                
                #print([kk,mm,trace_prho,purity_prho,renyi_1_prho])
            
                out.append([kk, mm, trace_prho, purity_prho, renyi_1_prho])
            
            
        return out
    
    
    
    ###########     ENTANGLEMENT        ################
    
    def compute_squeezing_inequalities_J (self):
        """
        Compute the left and right hand side of squeezing inequalities from Toth paper (PRA 89, 032307 (2014)), Eqs.(97)
        Use for that the first and second moments of collective spin.
        """
        jay = (self.numlevels-1)/2
        q0 = jay * (jay + 1) / 3
        
        Qmat = np.zeros((3,3))
        Cmat = np.zeros((3,3))
        gmat = np.zeros((3,3))
            
        xyz = ['x','y','z']
        for alpha in xyz:
            for beta in xyz:
                alpha_ind = xyz.index(alpha)
                beta_ind = xyz.index(beta)
                
                Qmat[alpha_ind,beta_ind] = 1/self.Ntotal * (np.trace( self.J2c[(alpha,beta)] @ self.rho)).real  -  q0 * self.kron_del(alpha,beta)
                Cmat[alpha_ind,beta_ind] = 1/2 * (np.trace( (self.Jc[alpha]@self.Jc[beta] + self.Jc[beta]@self.Jc[alpha]) @ self.rho)).real
                gmat[alpha_ind,beta_ind] = Cmat[alpha_ind,beta_ind] - (np.trace( self.Jc[alpha] @ self.rho)).real * (np.trace( self.Jc[beta] @ self.rho)).real
        
        Xmat = (self.Ntotal-1)*gmat + Cmat - self.Ntotal**2 * Qmat
        
        # Diagonalize Xmat
        Xevals = eigvalsh(Xmat)
        Xmin = np.amin(Xevals)
        Xmax = np.amax(Xevals)
        
        #print(Xevals)
        #print([Xmin,Xmax])
        
        #print(gmat)
        
        # Entanglement witnesses. State is entangled if any of these quantities is < 1.
        Ntot = self.Ntotal
        Njj = self.Ntotal*jay * (self.Ntotal*jay+1)
        NNj = self.Ntotal*(self.Ntotal-1)*jay
        
        ineq1 = Njj / np.trace(Cmat)
        ineq2 = np.trace(gmat) / (Ntot*jay)
        ineq3 = (Xmin+Njj) / ( np.trace(Cmat) + Ntot**2 * q0 )
        ineq4 = ( (Ntot-1)*np.trace(gmat) + Ntot**2 * q0 ) / ( Xmax + NNj )
        
        #print('Squeezing Js:')
        #print([ineq1,ineq2,ineq3,ineq4])
        
        return [ineq1,ineq2,ineq3,ineq4]
        
        
        
    def compute_squeezing_inequalities_GM (self):
        """
        Compute the left and right hand side of squeezing inequalities from Toth paper (PRL 107, 240502 (2011))
        Use generalized version to eigenvalues.
        Use for that the first and second moments of Gell-Mann matrices.
        """
        num_gen = len(self.gms)
        
        Qmat = np.zeros((num_gen,num_gen))
        Cmat = np.zeros((num_gen,num_gen))
        gmat = np.zeros((num_gen,num_gen))
            
        for alpha in range(num_gen):
            for beta in range(num_gen):
                
                Qmat[alpha,beta] = 1/self.Ntotal * (np.trace( self.GM2c[(alpha,beta)] @ self.rho)).real
                Cmat[alpha,beta] = 1/2 * (np.trace( (self.GMc[alpha]@self.GMc[beta] + self.GMc[beta]@self.GMc[alpha]) @ self.rho)).real
                gmat[alpha,beta] = Cmat[alpha,beta] - (np.trace( self.GMc[alpha] @ self.rho)).real * (np.trace( self.GMc[beta] @ self.rho)).real
        
        Xmat = (self.Ntotal-1)*gmat + Cmat - self.Ntotal**2 * Qmat
        
        print(Xmat)
        
        # Diagonalize Xmat
        Xevals = eigvalsh(Xmat)
        Xevals = np.sort(Xevals)
        
        #print(Xevals)
        
        # Right-hand side
        Kconst=1
        rhs = np.trace(Cmat) - self.Ntotal * np.trace(Qmat) - self.Ntotal*(self.Ntotal-1)*Kconst
        if param.hilberttype=='sym':
            if np.abs(rhs)>0.000001:
                print("\nERROR: Right-hand side of squeezing inequality is %g instead of zero.\n"%(rhs))
        
        # Entanglement witnesses. State is entangled if any of these quantities is < 0.
        ineq = []
        for ii in range(num_gen+1):
            ineq_ii = np.sum(Xevals[:ii]) - rhs
            ineq.append(ineq_ii)
        
        #print('Squeezing GM:')
        #print(ineq)
        
        return ineq
    
    
    
    def compute_concurrence (self):
        """
        WRONG CALCULATION. NEED TO MINIMIZE OVER RHO DECOMPOSITIONS --> VERY HARD.
        Diagonalize rho.
        Compute mixed state concurrence.
        """
        out = []
        
        # Diagonalize rho, output probabilities (ps), and states (psis)
        ps, psis = eig(self.rho)
        
        epsilon = 0.00001
        
        large_weights = ps.real > epsilon
        ps = ps[large_weights]
        psis = psis[:,large_weights]
        
        ordering = np.argsort(ps)[::-1]
        ps = ps[ordering]
        psis = psis[:,ordering]
        
        concurrence = 0
        
        eig_curr = -1
        
        for ii in range(len(ps)):
        
            # Compute Gell-Mann means
            GM_mean = [  np.squeeze( psis[:,ii].conj().T @ self.GMc[jj] @ psis[:,ii] ).real  for  jj in range(self.numlevels**2)  ]
    
            # Compute reduced 1-particle density matrix
            rho1 = np.zeros((self.numlevels,self.numlevels),dtype=complex)
    
            for jj in range(self.numlevels**2):
        
                lambda_j = GM_mean[jj]
                rho1 += lambda_j * self.gms[jj]

            # Compute single-particle concurrence
            concurrence_ii = np.sqrt( self.numlevels/(self.numlevels-1) * ( 1 - ((rho1@rho1).diagonal().sum()).real ) )
            concurrence += ps[ii].real * concurrence_ii
        
            print( [ii, ps[ii], concurrence_ii] )
                
            eig_curr = ps[ii]
        
        print(concurrence)
        
        """
        
        trace = (np.trace(self.rho)).real
        purity = (np.trace(self.rho@self.rho)).real
        
        # Compute Gell-Mann means
        GM_mean = [  ( (self.GMc[ii] @ self.rho).diagonal().sum() ).real  for  ii in range(self.numlevels**2)  ]
        
        # Compute reduced 1-particle density matrix
        rho1 = np.zeros((self.numlevels,self.numlevels),dtype=complex)
        
        for jj in range(self.numlevels**2):
            
            lambda_j = GM_mean[jj]
            rho1 += lambda_j * self.gms[jj]

        # Compute single-particle Renyi entropy
        renyi_1 = -np.log2( ((rho1@rho1).diagonal().sum()).real )
        
        out = [trace, purity, renyi_1]
        
        return out
        
        """
    
    
    def project_rho_ground (self):
        """
        Project rho onto ground manifold
        """
        
        out = []
        
        kk=0
        
            
        indices_kk = []
        #proj_kk = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        
        # Save indices of states with kk excitations
        for index in self.hspace.index_to_state:    # Run over all states
        
            state = self.hspace.index_to_state[index]
            
            if sum(state[param.deg_g:])==kk:   indices_kk.append(index)
    
        # Project rho
        projected_rho = self.rho[np.ix_(indices_kk,indices_kk)]
        
        # Normalize
        trace_prho = (np.trace(projected_rho)).real
        projected_rho = projected_rho / trace_prho
        
        # Purity
        purity_prho = (np.trace(projected_rho@projected_rho)).real
            
        return projected_rho
    
    
    '''
        
    def create_output_ops_excitedmanifold (self):
        """
        Creates operator for counting number of excitations.
        """
        self.op_excitedmanifold = csc_matrix((self.hspace.hilbertsize,self.hspace.hilbertsize))
        for aa in range(param.deg_e):
            for ii in range(self.Ntotal):
                sigma_i = self.hspace.sigma_matrix( self.eg_to_level['e'][aa] , self.eg_to_level['e'][aa] , ii )
                self.op_excitedmanifold = self.op_excitedmanifold + sigma_i
        self.op_excitedmanifold = self.op_excitedmanifold / self.Ntotal 
        
        
    
    '''

        













