# 

import numpy as np
#import cmath
#from cmath import exp as exp
from numpy import pi as PI
from numpy import exp as exp
from numpy import sin as sin
from numpy import cos as cos
from numpy import sqrt as sqrt

import itertools as it

import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix

import time
import sys


import parameters as param


class Hilbert_Space:
    
    """
    ----------------
    - Variables
    ----------------
    numlevels:          number of (internal) levels per site
    filling:            number of particles per site
    nstates:            number of states
    localhilbertsize:   
    hilbertsize:
    
    
    ----------------
    - Lists
    ----------------
    identities:         List of identities of different sizes
    localstates_list:        List of all states (in binary) of local Hilbert space
    binaries_list:      List of binary coding of localstates_list
    
    
    ----------------
    - Functions
    ----------------
    
    
    
    """
    
    def __init__(self,numlevels,filling,ntensorprods,hilberttype):
        
        self.numlevels = numlevels
        self.filling = filling
        self.ntensorprods = ntensorprods
        self.hilberttype = hilberttype
        
        # Local Hilbert space
        self.define_local_states()
        
        # Global Hilbert space
        if self.hilberttype=='full': self.define_global_states()
        if self.hilberttype=='sym': self.define_global_symmetric_states()

        self.compute_number_of_excitations (param.deg_g,param.deg_e)

        if param.output_stateslist: self.output_stateslist()
        
        print("Local Hilbert size: %i"%(self.localhilbertsize))
        print("Hilbert size: %i"%(self.hilbertsize))
            
        
        
    def store_identities(self):
        """
        Stores a list with (sparse) identity matrix of different sizes for fast construction of sigma_matrix operators.
        """
        self.identities = []
        for ii in range(self.ntensorprods):
            self.identities.append( sp.identity(self.localhilbertsize**ii) )
            
        
        
    def binary(self,state):
        """
        Expects array of 0s and 1s. Returns number represented by binary code given.
        """
        b = 0
        for ii in range(len(state)):
            b = b + state[ii]*2**ii
        return b
        
        
        
    def define_local_states (self):
        """
        Spins/F/B for filling==1
        Fermions for filling>1
        
        Creates and sorts in increasing order of binary hash number
        - localstates_list: list of all states in the local Hilbert space in binary coding.
        - binaries_list: contains the binary equivalent of the states.
        
        Note: Each state is represented by a list of 0 and 1, where a 1 in entry n means that the single-body state |n> is singly occupied.
        The many-body Fock states |(b0,b1,...)^T> are defined as (f_0^dagger)^b0 (f_1^dagger)^b1 ... |vacuum>
        """
        
        self.localstates_list = [np.zeros(self.numlevels, dtype=int)]
        self.binaries_list = []
        temp = []
        
        # Create list of states for a given number of particles per site
        for nn in range(1,self.filling+1):
            temp=self.localstates_list
            self.localstates_list=[]
            for ii in range(len(temp)):
                
                # Find last entry with a 1
                if nn>1: rindex = np.argwhere(temp[ii] == 1).max()
                else: rindex = 0
        
                for jj in range(rindex,self.numlevels):
                    b = temp[ii] + 0
                    b[jj] = b[jj] + 1
                    if b[jj]<=1: self.localstates_list.append(b)
        
        # Compute binaries and indices of all states
        for ii in range(len(self.localstates_list)):
            self.binaries_list.append(self.binary(self.localstates_list[ii]))
        
        # Order states in increasing order of binaries
        ordering = sorted(range(len(self.binaries_list)),key=self.binaries_list.__getitem__)
        self.localstates_list = [self.localstates_list[ii] for ii in ordering]
        self.binaries_list = [self.binaries_list[ii] for ii in ordering]
        
        self.localhilbertsize = len(self.localstates_list)
        
    
    
    def define_global_states (self):
        """
        The many-body states are defined as |n0,n1,...,n_{N-1}> for N atoms in computational basis, n_i in {0,...,numlevels-1}.
        Function creates dictionary of states:
        - index_to_state: dictionary with the index of all distinct many-body states.
        - state_to_index: Inverse dictionary.
        Assumes tensor product is defined as a x b = ( a_1*b , a_2*b, a_3*b, ... )
        """
        self.index_to_state = {}
        self.state_to_index = {}
        
        self.hilbertsize = self.localhilbertsize**self.ntensorprods
        
        Ntot = self.ntensorprods
        nlevels = self.numlevels
        
        # Loop through hilbertspace and get state in computational basis (could also do the opposite)
        for ii in range(self.hilbertsize):
            comp_basis = []
            y = ii
            m = self.hilbertsize/self.localhilbertsize
            index = int(y/m)
            comp_basis.append(index)
            for jj in range(1,self.ntensorprods):
                y = y - index*m
                m = m/self.localhilbertsize
                index = int(y/m)
                comp_basis.append(index)
            
            self.index_to_state[ii] = tuple(comp_basis)
            self.state_to_index[tuple(comp_basis)] = ii
            
        print("index_to_state:")
        print(self.index_to_state)
        print()
        print("state_to_index:")
        print(self.state_to_index)
        print()
    
    
    def define_global_symmetric_states (self):
        """
        Permutationally symmetric states are defined by a vector m with m_i giving the number of atoms in state m_i.
        This function makes a dictionary of all distinct states |m0,m1,...,m_nlevels> and assigns each of them a number.
        Inverse dictionary is also produced.
        Note: first state in list is (0,0,...,N) instead of (N,0,0...)
        """
        Ntot = self.ntensorprods
        nlevels = self.numlevels

        combinations = it.combinations(range(Ntot+nlevels-1), nlevels-1)
        
        def bars_to_populations(bars):
            bars = (-1,) + bars + (Ntot+nlevels-1,)
            bars = np.array(bars)
            return tuple( bars[1:] - bars[:-1] - 1 )

        self.index_to_state = { idx : bars_to_populations(bars)
                           for idx, bars in enumerate(combinations) }

        self.state_to_index = { state : idx for idx, state in self.index_to_state.items() }
        
        self.hilbertsize = len(self.index_to_state)
        
        print("index_to_state:")
        print(self.index_to_state)
        print()
        print("state_to_index:")
        print(self.state_to_index)
        print()
        
    
    def compute_number_of_excitations (self,eindex,nestates):
        """
        Assuming that the internal levels are ordered as [ground levels, excited levels, other levels],
        compute and save for each state in localstates_list the number of excitations present between eindex and eindex+nestates.
        """
        # Save list of excitations of each state in local Hilbert space
        self.local_excitations_list = []
        for ii in range(len(self.localstates_list)):
            self.local_excitations_list.append( sum(self.localstates_list[ii][eindex:eindex+nestates]) )
            
        # Compute and save number of excitations for each state in Hilbert space
        self.state_to_number_excitations = {}
        self.index_to_number_excitations = {}
        for state in self.state_to_index:
            
            if self.hilberttype=='full':
                self.state_to_number_excitations[state] = np.sum( np.asarray( [ self.local_excitations_list[state[jj]] for jj in range(len(state)) ] , dtype=int ) )
                
            if self.hilberttype=='sym':
                self.state_to_number_excitations[state] = np.sum( np.asarray( [ self.local_excitations_list[nn]*state[nn] for nn in range(len(state)) ] , dtype=int ) )
                
            self.index_to_number_excitations[self.state_to_index[state]] = self.state_to_number_excitations[state]
        
        print("state_to_number_excitations")
        print(self.state_to_number_excitations)
        print()
        
        
        
    def sigma_matrix (self,alpha,beta,ii):
        """
        Outputs (sparse) matrix representation of operator sigma_alphabeta^(ii) = f_alpha^dagger^(ii) f_beta^(ii) for current Hilbert space
        """
        if self.filling==1:
            # This doesn't seem to be significantly faster than general version
            return sp.kron( sp.kron( sp.identity(self.localhilbertsize**ii) , self.transitionOp_local( alpha, beta ) , format='csc'), sp.identity(self.localhilbertsize**(self.ntensorprods-ii-1)) , format='csc')
        else:
            return sp.kron( sp.kron( sp.identity(self.localhilbertsize**ii) , self.sigma_matrix_local( alpha, beta ) , format='csc'), sp.identity(self.localhilbertsize**(self.ntensorprods-ii-1)) , format='csc')
        
        # Storing identities not much faster, apparently.
        #return sp.kron( sp.kron( self.identities[ii] , self.sigma_matrix_local( alpha , beta ) , format='csc'), self.identities[self.ntensorprods-ii-1] , format='csc')
        
        
        
    def sigma_matrix_local (self,alpha,beta):
        """
        Fermions
        
        Outputs (sparse) matrix representation of operator sigma_alphabeta = f_alpha^dagger f_beta for current local Hilbert space
        """
        states = np.array(self.localstates_list)
        indices = np.arange(len(self.localstates_list))
        binaries = np.array(self.binaries_list)
        
        # Apply f_beta to states and remove previously unoccupied states
        states[:,beta] = states[:,beta] - 1
        
        remove = states[:,beta]>=0
        states = states[remove,:]
        indices = indices[remove]
        binaries = binaries[remove]
        
        # Apply f_alpha^dagger to states and remove previously occupied states
        states[:,alpha] = states[:,alpha] + 1
        
        remove = states[:,alpha]<=1
        states = states[remove,:]
        indices = indices[remove]
        binaries = binaries[remove]
        
        # Compute sign of operation
        if abs(alpha-beta)>1:
            if alpha<beta: sign = np.sum(states[:,alpha+1:beta],axis=1)
            else: sign = np.sum(states[:,beta+1:alpha],axis=1)
        else: sign = np.zeros(len(states), dtype=int)
            
        sign = (-1)**sign
        
        # Compute binaries of resulting states and substitute by their state number
        states = self.binary(states.transpose())
        states = np.searchsorted(self.binaries_list,states)
        
        #for ii in range(len(states)):
        #    print(indices[ii],states[ii],self.localstates_list[indices[ii]],self.localstates_list[states[ii]],sign[ii])
        
        # Compute sigma (sparse matrix)
        sigma = csc_matrix( (sign,(states,indices)) , shape=(self.localhilbertsize,self.localhilbertsize) )
        #sigma = np.zeros((self.localhilbertsize,self.localhilbertsize))
        #for ii in range(len(states)):
        #    sigma[states[ii],indices[ii]] = sign[ii]
        
        #print(sigma.toarray())
        
        return sigma
        
        
        
    def transitionOp_onsite (self,n1,n2,ii):
        """
        Outputs (sparse) matrix representation of transition operator |n1><n2|_ii, where n1, n2 label states in the local Hilbert space ii
        """
        return sp.kron( sp.kron( sp.identity(self.localhilbertsize**ii) , self.transitionOp_local(n1,n2) , format='csc'), sp.identity(self.localhilbertsize**(self.ntensorprods-ii-1)) , format='csc')
    
    
    def transitionOp_sym (self,state1,state2):
        """
        Returns global transition operator |state1><state2|.
        state1, state2 are expected to be integer lists (or tuples) of length numlevels, where each entry of state1 (state2) gives number of atoms in that level.
        """
        n1 = self.state_to_index[tuple(state1)]
        n2 = self.state_to_index[tuple(state2)]
        return self.transitionOp_global(n1,n2)
            
            
    def transitionOp_local (self,n1,n2):
        """
        Returns local transition operator |n1><n2|, where n1, n2 label states in the local Hilbert space
        """
        return csc_matrix( ([1],([n1],[n2])) , shape=(self.localhilbertsize,self.localhilbertsize))
        
        
    def transitionOp_global (self,n1,n2):
        """
        Returns global transition operator |n1><n2|, where n1, n2 label states in the global Hilbert space
        """
        return csc_matrix( ([1],([n1],[n2])) , shape=(self.hilbertsize,self.hilbertsize))
        
        
        
    def sigma_ab_sym (self,aa,bb):
        """
        Returns collective sigma_ab = sum_i |a><b|_i in permutationally symmetric manifold.
        States aa and bb should be integers labeling the internal atomic levels.
        """
        sigma = lil_matrix( (self.hilbertsize,self.hilbertsize) )
        
        changestate = np.zeros(self.numlevels, dtype='int')
        changestate[aa] += 1
        changestate[bb] += -1
        
        for state in self.state_to_index:
            rstate = np.array( state , dtype='int' )
            lstate = rstate + changestate
            if tuple(lstate) in self.state_to_index:
                sigma[ self.state_to_index[tuple(lstate)] , self.state_to_index[tuple(rstate)] ] += sqrt( lstate[aa] * rstate[bb] )
        
        return sigma
        
    
    
    def get_statenumber(self,m):
        """
        Expects 1D array of integers specifying the levels that are occupied by an atom, i.e. | (m[0], m[1], m[2], ...) >  (ordered)
        Returns state number corresponding to that state.
        
        Note: Should add more exceptions and error handling.
        """
        b = np.zeros(self.numlevels, dtype=int)
        for ii in range(len(m)):
            if m[ii]<len(b):
                b[m[ii]] = 1
            else: print("Error: get_statenumber: Level numbers given larger than available.")
        if np.sum(b)!=self.filling: print("Error: get_statenumber: Too few levels given.")
        b = self.binary(b)
        return np.searchsorted(self.binaries_list,b)
        

    def output_stateslist(self):
        """Outputs file with the state corresponding to each statenumber.
        Move this to cavity_ED.py??"""
        ## Output list of states of local Hilbert space
        filename = '%s/liststates_fill%s_Ng%i_Ne%i_Ni%i.txt'%(param.outfolder, param.filling, param.deg_g, param.deg_e, param.deg_i)
        with open(filename,'w') as f:
            f.write('# Row 1: state number | Row 2: occupation excited | Row 3: occupation intermediate (if any) | Row 4: occupation ground.\n')
            f.write('# \n# \n')
            for ii in range(len(self.localstates_list)):
                f.write('%i\n'%(ii))
                np.savetxt(f,[list(self.localstates_list[ii][param.deg_g:param.deg_g+param.deg_e])],fmt='%i')
                if param.deg_i>0: np.savetxt(f,[list(self.localstates_list[ii][param.deg_g+param.deg_e:])],fmt='%i')
                np.savetxt(f,[list(self.localstates_list[ii][:param.deg_g])],fmt='%i')
                f.write('\n')
                
                #output_data = [ [ tt ] + list(self.localstates_list[tt]) for tt in range(len(self.localstates_list)) ]
                #np.savetxt(f,output_data,fmt='%i')
        
        
        ## Output |x> states of total Hilbert space in computational basis |y1,y2,y3,..>, where each y corresponds to a state of self.localstates_list
        #if self.ntensorprods>1:
        #    filename = '%s/globalstates_ntensorprods%i_fill%s_Ng%i_Ne%i.txt'%(param.outfolder, self.ntensorprods, param.filling, param.deg_g, param.deg_e)
        #    with open(filename,'w') as f:
        #        f.write('# Row 1: global state number |x> | Row 2: stat in computational basis |y1,y2,y3,..>\n')
        #        f.write('# \n# \n')
        #        for ii in range(self.hilbertsize):
        #            f.write('%i\n'%(ii))
        #            # Write hash in computational basis
        #            y = ii
        #            m = self.hilbertsize/self.localhilbertsize
        #            index = int(y/m)
        #            f.write('%i '%(index))
        #            for jj in range(1,self.ntensorprods):
        #                y = y - index*m
        #                m = m/self.localhilbertsize
        #                index = int(y/m)
        #                f.write('%i '%(index))
        #            f.write('\n\n')
    











