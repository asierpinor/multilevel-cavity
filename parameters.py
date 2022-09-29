
import numpy as np
from numpy import pi as PI
from numpy import sin as sin
from numpy import cos as cos

import math
from math import sqrt as sqrt

from sympy import S

import sys


# -------------------------------------------
#           System parameters
# -------------------------------------------

"""
Units:
    Lengths in micrometers
    Frequencies in 1/microsecond
    Decay rate in 1/microsecond
"""
rescale = True     # Rescale frequencies with Gamma and lengths with lambda0

# Nature constants
c = 2.99*10**8       # Speed of light  micrometer/microsecond

### ------------------------
###         Method
### ------------------------
method = 'ED'     # ED: exact diagonalization
                        # MF: mean-field
                        # TWA: mean-field + sampling
                        # cumulant: cumulant to second order
                        # bbgky: cumulant to second order + sampling
subleading_terms = ''       # 'sub' to activate subleading terms in eom
                            # 'selfint' to activate self-interaction terms in MF eom
iterations = 1000          # Number of classical trajectories for twa
sampling = 'gaussian'   # gaussian: continuous gaussian approximation to Wigner
                        # discrete: generalized discrete TWA from Bihui and Johannes


### ------------------------
###     Geometry settings
### ------------------------
geometry = 'alltoall'        # Choose ensemble geometry: alltoall
Ntotal = 2     # Total number of atoms in cavity
filling = 1     # Filling per site


### ------------------------
###     Hilbert space
### ------------------------
hilberttype = 'sym'        # Possible truncations of Hilbert space. 'full': all states, 'sym': permutationally symmetric states.


### ---------------------
###     Atom properties
### ---------------------
Fe=S(3)/2                   # Note: Must use symbolic numbers S() from sympy
Fg=S(1)/2
deg_e=2
deg_g=2
#deg_e=int(2*Fe+1)          # Degeneracy of upper levels
#deg_g=int(2*Fg+1)        # Degeneracy of lower levels
start_e=2               # Levels go from -Fe+start_e until -Fe+start_e+deg_e
start_g=0               # Same for g

Gamma = 1 #0.04712                     # Decay rate of excited state (1/microsecond)
lambda0 = 0.689                     # Transition wave-length (micrometers)
k0 = 2*PI/lambda0                   # wave-vector of dipole transition (1/micrometer)
omega0 = c*k0         # frequency between levels A and B  (1/microsecond)

zeeman_e = 0*Gamma        # Constant energy shift between |e mu> and |e mu+1> (1/microsecond)
zeeman_g = 0*Gamma        # Constant energy shift between |g mu> and |g mu+1> (1/microsecond)

gamma0 = 0*Gamma           # Single-particle spontaneous emission

epsilon_ne = 0*Gamma            # Add a constant epsilon_ne* sum_m sigma_{e_m e_m} such that eigenvalues of different n_e are not degenerate
epsilon_ne2 = 0*Gamma            # Add a constant epsilon_ne* sum_m sigma_{e_m e_m} such that n_e=2 are off-resonant

clebsch_zero_q = [0,-1]             # Set all Clebsch-Gordan coefficients with a given "q" to zero, i.e. 0,1, or -1.
#clebsch_zero_q = []

### ------------------------
###     Cavity properties
### ------------------------
g_coh = 0*Gamma/Ntotal       # Effective atom-atom coupling (incoherent)
#g_coh = 0*Gamma
g_inc = 2*Gamma/Ntotal           # Effective atom-atom coupling (coherent)
#g_inc = 0*Gamma*(3/2)

#zeeman_e = -0*g_coh*Ntotal/2


### ----------------------------
###     Pump Laser properties
### ----------------------------
rabi_Pi = 0*Gamma*sqrt(3)/2 #*Ntotal / 2
rabi_Sigma = 0.187/2*sqrt(2)*Gamma     # 0.3, 0.35, 0.38, 0.388, 0.388619
detuning = 0*Gamma      # Laser detuning with respect to atomic transition. detuning = omega_laser-omega0

########## Managing the lasers ##########
switchtimes = [0]       # Time step where switching happens
switchlaser = [False]       # True: Switch lasers on, False: Switch lasers off

will_laser_be_used = False
for ii in range(len(switchlaser)):
    if switchlaser[ii]==True: will_laser_be_used = True


### ------------------------
###     Initial conditions
### ------------------------
#cIC = 'test'
cIC = 'puresupstate'     # Choose initial condition:
                    # 'initialstate': All atoms in state defined by 'initialstate'
                    # 'mixedstate': All atoms in mixed state of ground states with probabilities given by mixed_gs_probabilities
                    # 'puresupstate': All atoms in pure superposition state of ground states with amplitudes given by pure_gs_amplitudes
                    # 'byhand': Hard code some state.
initialstate = ['g0']
mixed_es_probabilities = [0.5,0]      # Probabilities of excited-levels for mixed state.
mixed_gs_probabilities = [0,0.5]      # Probabilities of ground-levels for mixed state.
pure_es_amplitudes = [0,0]       # Amplitudes of excited-levels for pure superposition state
pure_gs_amplitudes = [1/sqrt(2),-1/sqrt(2)]       # Amplitudes of ground-levels for pure superposition state

rotate = True      # Decide if apply rotation to initial state with H_Rabi and Rabi frequencies Omet_Pi and Omet_Sigma
relevantClebsch = 1/sqrt(2)          # 1/2,1/2, Pi: 1/sqrt(3) Sigma: sqrt(2/3) // 1,1: Pi: 1/sqrt(2) Sigma: 1/2 // 0,1: Pi,Sigma: 1
                                      # 3/2,3/2: Pi: 3/sqrt(15), 1/sqrt(15) // 9/2,9/2: 1/(3*sqrt(11)) for Pi 1/2.
                                      # 1/2,3/2: Pi: sqrt(2/3) Sigma: 1/sqrt(3)
Omet_Pi = PI*0 / (2*relevantClebsch)       # Value of Omega_Pi * tau
Omet_Sigma = -PI*4.074 / (2*relevantClebsch)    # Value of Omega_Sigma * tau             #3.873, 2.123    # 4.218, 4.307, 4.395, 4.493   # 4.074

### ------------------------------------
###         Second phase of dynamics
### ------------------------------------

phase2 = False
Ntstart_phase2 = 200        # At which time do we apply rotation and change parameters to continue in phase 2.
# Initial rotation
Omet_Pi_phase2 = PI*0 / (2*relevantClebsch)       # Value of Omega_Pi * tau
Omet_Sigma_phase2 = PI*0 / (2*relevantClebsch)    # Value of Omega_Sigma * tau             #3.873, 2.123    # 4.21875, 4.30777, 4.39562, 4.472

rabi_Pi_phase2 = 0*Gamma*sqrt(3)/2 #*Ntotal / 2
rabi_Sigma_phase2 = 0/2*sqrt(2)*Gamma     # 0.388619



### ------------------------
###         Other
### ------------------------
digits_trunc = 6        # Number of digits to truncate when computing excitation manifold of eigenstates
      
### ------------------------
###         Output
### ------------------------
bin_factor = 100            # TWA output of fullDistribution will be binned with iterations/bin_factor bins from 0 to 1.
#outfolder = './data'
#outfolder = '.'
#outfolder = '../Simulations/Superradiance_experiment'
#outfolder = '../Simulations/MultiSuperradiance_experiment'
#outfolder = '../Simulations/Chi_dynamics_experiment'
#outfolder = '../Simulations/Robert_Dicke_preparation'
#outfolder = '../Simulations/Paper_dynTransition'
#outfolder = '../Simulations/Paper_superradiance_6l_8l'
#outfolder = '../Simulations/Paper_qfluctuations'
#outfolder = '../Simulations/Paper_darks_6l'
#outfolder = '../Simulations/Eigenstates/Ng%i_Ne%i'%(deg_g,deg_e)
#outfolder = '../Simulations/Eigenvalues_Paper'
#outfolder = '../Simulations/Benchmarks/N1_MF_vs_exact'
#outfolder = '../Simulations/Benchmarks'
outfolder = '../Simulations/Squeezing'
#outfolder = '../Simulations/Edwin'
#outfolder = '../../../../../Dropbox/Andre_Asier/Benchmark_codes'

#which_observables = ['populations','pop_variances','xyz']
which_observables = ['populations','xyz','photons','xyz_variances','gell_mann']
#which_observables = ['populations','photons']
                                        # populations: output only |nn><nn|
                                        # xyz: output pauli_x, pauli_y and pauli_z for each pair of states
                                        # photons: output PiPi, SigmaSigma, PiSigma.real, PiSigma.imag
                                        # xyz_variances: sx^2, sy^2, sz^2, sxsy, sxsz, sysz (symmetrized)
                                        # gell_mann: lambda_i and (lambda_i lambda_j + lambda_j lambda_i)/2 for all Gell-Mann matrices lambda_i
                                        # fisher: Compute quantum fisher information from maximum eigenvalue of Gamma_ij
output_occupations = True
output_eigenstates = False
output_dark_renyi = False
output_stateslist = False
output_GSdistribution = False
output_fullDistribution = False
output_purity_renyi = False
output_squeezing = False
append = ''
#append='_clebschzeroQ%s_ze%g'%(''.join( [ str(clebsch_zero_q[aa]) for aa in range(len(clebsch_zero_q)) ] ), zeeman_e/Gamma)
#append = '_Fg%gFe%g_Sig%g'%(float(Fg),float(Fe),Omet_Sigma/PI*2*relevantClebsch/sqrt(2))
#append = '_Fg%gFe%g_starte%i_Sig%g'%(float(Fg),float(Fe),start_e,Omet_Sigma/PI*2*relevantClebsch)
#append = '_Fg%gFe%g_starte%i_quick'%(float(Fg),float(Fe),start_e)
#append = '_Fg%gFe%g_startg%i_starte%i'%(float(Fg),float(Fe),start_g,start_e)
#append = '_Fg%gFe%g_startg%i_starte%i_Sig%g'%(float(Fg),float(Fe),start_g,start_e,Omet_Sigma/PI*2*relevantClebsch)
#append = '_Fg%gFe%g_stretchSuperp'%(float(Fg),float(Fe))
#append = '_Fg%gFe%g_Pidyn2'%(float(Fg),float(Fe))
#append = '_Fg%gFe%g'%(float(Fg),float(Fe))
#append = '_Fg%gFe%g_CircularPlus_Pi%gSig%g_DistAna'%(float(Fg), float(Fe), Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch)
#append = '_Fg%gFe%g_chi%g_Gamma%g_CircularPlus_Pi%gSig%g_squeezAna'%(float(Fg), float(Fe), g_coh/Gamma*Ntotal,g_inc/Gamma*Ntotal, \
#                                                                    Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch)
#append = '_Fg%gFe%g_chi%g_Gamma%g_CircularPlus_Pi%gSig%g'%(float(Fg), float(Fe), g_coh/Gamma*Ntotal,g_inc/Gamma*Ntotal, \
#                                                                    Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch)
#append = '_Fg%gFe%g_chi%g_Gamma%g_paritySigma_Pi%gSig%g'%(float(Fg), float(Fe), g_coh/Gamma*Ntotal,g_inc/Gamma*Ntotal, \
#                                                                    Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch)
#append = '_Fg%gFe%g_chi%g_Gamma%g_CircularPlus_Pi%gSig%g'%(float(Fg), float(Fe), g_coh/Gamma*Ntotal,g_inc/Gamma*Ntotal, \
#                                                                    Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch)
if phase2 == False:
    append = '_Fg%gFe%g_chi%g_Gamma%g_CircularPlus_Pi%gSig%g_rabiPi%grabiSig%g'%(float(Fg), float(Fe), g_coh/Gamma*Ntotal,g_inc/Gamma*Ntotal, \
                                                                    Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch, rabi_Pi/Gamma/sqrt(2)*2, rabi_Sigma/Gamma/sqrt(2)*2)
if phase2 == True:
    append = '_Fg%gFe%g_chi%g_Gamma%g_CircularPlus_Pi%gSig%g_rabiPi%grabiSig%g_ph2Pi%gSig%g_rabiPi%grabiSig%g'%(float(Fg), float(Fe), g_coh/Gamma*Ntotal,g_inc/Gamma*Ntotal, \
                                                                    Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch, rabi_Pi/Gamma/sqrt(2)*2, rabi_Sigma/Gamma/sqrt(2)*2, \
                                                                    Omet_Pi_phase2/PI*2*relevantClebsch, Omet_Sigma_phase2/PI*2*relevantClebsch, rabi_Pi_phase2/Gamma/sqrt(2)*2, rabi_Sigma_phase2/Gamma/sqrt(2)*2 )
#append = '_Fg%gFe%g_chi%g_Gamma%g_RLphase_Pi%gSig%g'%(float(Fg), float(Fe), g_coh/Gamma*Ntotal, g_inc/Gamma*Ntotal, rabi_Pi/Gamma, rabi_Sigma/Gamma)
#substr = ''
#if subleading_terms=='selfint': substr = '_selfint'
#append = '_Fg%gFe%g_CircularPlus_Pi%gSig%g_gamma%g%s'%(float(Fg), float(Fe), Omet_Pi/PI*2*relevantClebsch, Omet_Sigma/PI*2*relevantClebsch, gamma0/Gamma, substr)

#append = '_Fg%gFe%g_p%s'%(float(Fg),float(Fe),'-'.join([ str(mixed_gs_probabilities[ii]) for ii in range(len(mixed_gs_probabilities)) ]) \
#                                            +'-'+'-'.join([ str(mixed_es_probabilities[ii]) for ii in range(len(mixed_es_probabilities)) ]))
#append = '_Pi%gSig%g'%(Omet_Pi/PI*2*relevantClebsch,Omet_Sigma/PI*2*relevantClebsch)
#append = '_chi%g_Gamma%g_Pi%gSig%g'%(g_coh/Gamma*Ntotal,g_inc/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch,Omet_Sigma/PI*2*relevantClebsch)
#append = '_Pi%gSig%g'%(Omet_Pi/PI*2*relevantClebsch,Omet_Sigma/PI*2*relevantClebsch)
#append = '_chi%g_Pi%gSig%g'%(g_coh/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch,Omet_Sigma/PI*2*relevantClebsch)
#append = '_Ome%gDet%g_chi%g_Pi%gSig%g'%(rabi_Pi/Gamma,detuning/Gamma,g_coh/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch,Omet_Sigma/PI*2*relevantClebsch)
#append = '_zg%gze%g_chi%g_Pi%gSig%g'%(zeeman_g/Gamma,zeeman_e/Gamma,g_coh/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch,Omet_Sigma/PI*2*relevantClebsch)
#append = '_OmePi%gOmeSig%gDet%g_zg%gze%g_chi%g_Pi%gSig%g'%(rabi_Pi/Gamma,rabi_Sigma/Gamma,detuning/Gamma,zeeman_g/Gamma,zeeman_e/Gamma,g_coh/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch,Omet_Sigma/PI*2*relevantClebsch)
#append = '_chi%g_Pirot%g'%(g_coh/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch)
#append = '_Ome%g_chi%g_Pirot%g'%(rabi_Pi/Gamma,g_coh/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch)
#append = '_Det%g_Ome%g_chi%g_Pirot%g'%(detuning/Gamma,rabi_Pi/Gamma,g_coh/Gamma*Ntotal,Omet_Pi/PI*2*relevantClebsch)

### ------------------------
###         Solver
### ------------------------
#solver = 'exp'         # Solve dynamics by exponentiating Linblad
solver = 'ode'          # Solve system of differential equations
atol = 1E-12            # Absolute tolerance for solver. Default: 1E-12
rtol = 1E-6             # Relative tolerance for solver. Default: 1E-6


### ------------------------
###     Time evolution
### ------------------------
dt=0.5/Gamma
Nt=200 #200
max_memory=0.4        # Maximal memory allowed in Gb. If more needed, abort program.






###
### RESCALE PARAMETERS WITH GAMMA AND LAMBDA0
###

"""
Only dimensionless quantities:

Frequency = Frequency / Gamma
Length = Length / lambda0

Note that omega = c*k ---->  omega = c/(lambda0*Gamma) k , where omega and k are rescaled with Gamma and lambda0
"""

if rescale == True:
    
    c = c/(lambda0*Gamma)   # Frequency*Length
    
    k0 = k0*lambda0                   # 1/Length
    
    omega0 = omega0/Gamma            # Frequency
    zeeman_g = zeeman_g/Gamma        # Frequency
    zeeman_e = zeeman_e/Gamma        # Frequency
    epsilon_ne = epsilon_ne/Gamma       # Frequency
    gamma0 = gamma0/Gamma               # Frequency
    
    rabi_Pi = rabi_Pi/Gamma                 # Frequency
    rabi_Sigma = rabi_Sigma/Gamma           # Frequency
    detuning = detuning/Gamma               # Frequency
    
    rabi_Pi_phase2 = rabi_Pi_phase2/Gamma                 # Frequency
    rabi_Sigma_phase2 = rabi_Sigma_phase2/Gamma           # Frequency
    
    dt=dt*Gamma
    
    g_inc = g_inc/Gamma
    g_coh = g_coh/Gamma
    
    Gamma=1
    lambda0=1




###
### CHECKS
###

if method not in ['ED','MF','TWA','cumulant','bbgky']:
    print('\nERROR/parameters: method chosen not valid.\n')
    sys.exit()
    
    
if sampling not in ['gaussian']:
    print('\nERROR/parameters: sampling chosen not valid.\n')
    sys.exit()
    

if deg_g>2*Fg+1 or deg_e>2*Fe+1:
    print('\nERROR/parameters: number of levels larger than degeneracy of F.\n')
    
if start_e+deg_e>2*Fe+1 or start_g+deg_g>2*Fg+1:
    print('\nERROR/parameters: number chosen for start_g/e larger than allowed by Fg/Fe.\n')

check_numberSw = [len(switchtimes),len(switchlaser)]
if check_numberSw.count(check_numberSw[0]) != len(check_numberSw):
    print('\nERROR/parameters: Length of switch parameter arrays inconsistent.\n')


if cIC == 'initialstate':
    if len(initialstate) != filling:
        print('\nERROR/parameters: Initial state specified incorrect length for chosen filling.\n')
        
        
if epsilon_ne!=0:
    print('\nINFO/parameters: bias term epsilon_ne is nonzero.\n')


if cIC=='mixedstate':
    if len(mixed_gs_probabilities)!=deg_g:
        print('\nERROR/parameters: length of probabilities vector for mixedstate not correct.\n')
    if len(mixed_es_probabilities)!=deg_e:
        print('\nERROR/parameters: length of probabilities vector for mixedstate not correct.\n')
    if abs(sum(mixed_gs_probabilities)+sum(mixed_es_probabilities)-1)>0.000000001:
        print('\nERROR/parameters: Sum of probabilities for mixedstate is not 1.\n')


if cIC=='puresupstate':
    if len(pure_gs_amplitudes)!=deg_g:
        print('\nERROR/parameters: length of amplitudes vector for puresupstate not correct.\n')
    if len(pure_es_amplitudes)!=deg_e:
        print('\nERROR/parameters: length of amplitudes vector for puresupstate not correct.\n')
    if abs( sum( [abs(pure_gs_amplitudes[ii])**2 for ii in range(len(pure_gs_amplitudes))] )\
             + sum( [abs(pure_es_amplitudes[ii])**2 for ii in range(len(pure_es_amplitudes))] ) - 1 )>0.000000001:
        print('\nERROR/parameters: Norm of state in puresupstate is not 1.\n')
    
    
    






