# 

import numpy as np
import matplotlib.pyplot as plt
#import cmath
#from cmath import exp as exp
from numpy import pi as PI
from numpy import exp as exp
from numpy import sin as sin
from numpy import cos as cos

from sympy.physics.quantum.cg import CG
from sympy import S

#from scipy.optimize import curve_fit

import time
import sys


import parameters as param
import cavity_ED
import cavity_MFcumulant


"""
Explain code here


"""



# -------------------------------------------
#           Functions
# -------------------------------------------
        
        
# Prints execution time in hours, minutes and seconds
def print_time(t):
    hours = t//3600;
    minutes = (t-3600*hours)//60;
    seconds = (t-3600*hours-60*minutes);
    print("Execution time: %i hours, %i minutes, %g seconds"%(hours,minutes,seconds));






# -------------------------------------------
#           Set up system
# -------------------------------------------

starttime = time.time()
lasttime = starttime




print("\n-------> Setting up system\n")

if param.method=='ED':
    
    print("\nMethod: ED\n")
    cavsys = cavity_ED.Cavity_ED()

    if param.Nt>0:
        if cavsys.memory_estimate_sparse_Linblad>param.max_memory:
            print("Memory estimated is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()

    if param.output_eigenstates:
        if cavsys.memory_full_Hamiltonian>param.max_memory:
            print("Memory of full Hamiltonian is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()
    

if param.method in ['MF','TWA','cumulant','bbgky']:
    
    print("\nMethod: %s\n"%(param.method))
    cavsys = cavity_MFcumulant.Cavity_MFcumulant()
    
    if param.Nt>0:
        if cavsys.memory_variables>param.max_memory:
            print("Memory estimated is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()











# -------------------------------------------
#           Compute eigenvectors
# -------------------------------------------

print("\n-------> Computing eigenvectors\n")


if param.method=='ED' and ( param.output_eigenstates or param.output_dark_renyi ):
    
    cavsys.save_only_H_cavity()
    
    if param.output_eigenstates:
        cavsys.compute_eigenstates()
    if param.output_dark_renyi:
        darksRenyi_data = cavsys.compute_Renyi_darkstates_fixed_k_m()

    print("\nTime computing eigenvectors.")
    print_time(time.time()-lasttime)
    lasttime = time.time()
    
else: print("No.")




# -------------------------------------------
#                 Dynamics
# -------------------------------------------

print("\n-------> Computing dynamics\n")



if param.Nt>0:

    # --------
    ##########  Set up initial time
    # --------

    phase = 0       # Traces stage of evolution, if Hamiltonian has sudden changes in time, e.g. laser switch on/off

    cavsys.choose_initial_condition()

    if param.method=='ED':
    
        cavsys.save_HamLin()
        cavsys.define_linblad_superop(phase)

        cavsys.compute_memory()
        if cavsys.memory_sparse_Linblad+cavsys.memory_sparse_Hamiltonian > param.max_memory:
            print("Memory is above maximal allowed memory of %g Gb.\nAbort program"%(param.max_memory))
            sys.exit()

        if param.solver == 'exp': cavsys.compute_evolution_op()
        if param.solver == 'ode': cavsys.set_solver(0)


        print("\nTime for constructing Linblad at t=0.")
        print_time(time.time()-lasttime)
        lasttime = time.time()
        
        
    if param.method in ['MF','TWA','cumulant','bbgky']:
        
        cavsys.set_solver(0)
    

    # --------
    ##########  Evolve
    # --------
    
    
    
    # Prepare output
    times=[ 0 ]
    if param.output_occupations:
        
        if param.method=='ED':
            cavsys.create_output_ops_occs()
            if param.output_purity_renyi == True:
                cavsys.save_initial_rho()
                out_purity_renyi = [ cavsys.compute_purity_and_Renyi() ]
            
            if param.output_squeezing == True:
                out_squeezing_J = [ cavsys.compute_squeezing_inequalities_J() ]
                out_squeezing_GM = [ cavsys.compute_squeezing_inequalities_GM() ]
                
        out_occup_states = [ cavsys.read_occs() ]

    # Time evolution
    for tt in range(1,param.Nt+1):
        
        print(times[tt-1])
        
        # Evolve one step
        if param.method=='ED':
            if param.solver == 'ode':
                if cavsys.solver.successful():
                    cavsys.evolve_rho_onestep()
                else: print("\nERROR: Problem with solver, returns unsuccessful.\n")
                times.append(cavsys.solver.t)
    
            if param.solver == 'exp':
                cavsys.evolve_rho_onestep()
                times.append( tt*param.dt )
        if param.method in ['MF','TWA','cumulant','bbgky']:
            if cavsys.solver.successful():
                cavsys.evolve_onestep()
            else: print("\nERROR: Problem with solver, returns unsuccessful.\n")
            times.append(cavsys.solver.t)
            
            
        # Change values of parameters
        if param.phase2:
            if tt == param.Ntstart_phase2:
                if param.method=='ED':
                    # Apply quick rotation pulse
                    cavsys.rotate_pulse(param.Omet_Pi_phase2,param.Omet_Sigma_phase2)
                    
                    # Change parameters
                    cavsys.redefine_H_rabi(param.rabi_Pi_phase2,param.rabi_Sigma_phase2)
                    cavsys.define_linblad_superop(1)    # phase variable has no meaning for now
                    
                    # Update solver
                    if param.solver == 'exp': cavsys.compute_evolution_op()
                    if param.solver == 'ode': cavsys.set_solver(cavsys.solver.t)
                
        #if len(param.switchtimes)>phase+1:
            #if param.switchtimes[phase+1]==tt:
                #phase = phase + 1
                #if param.method=='ED':
                    #cavsys.define_linblad_superop(phase)
                    #if param.solver == 'exp': cavsys.compute_evolution_op()
                    #if param.solver == 'ode': cavsys.set_solver(cavsys.solver.t)
                    
        
        # Save observables
        if param.output_occupations: out_occup_states.append(cavsys.read_occs())    
        if param.method=='ED' and param.output_purity_renyi == True:
            out_purity_renyi.append( cavsys.compute_purity_and_Renyi() )
        if param.method=='ED' and param.output_squeezing == True:
            out_squeezing_J.append( cavsys.compute_squeezing_inequalities_J() )
            out_squeezing_GM.append( cavsys.compute_squeezing_inequalities_GM() )
        
        
        
        
    
    #print(times[tt-1])
    #cavsys.compute_purity_and_Renyi_of_fixed_k_m()
        
        
    # Save Renyi entropy and purity for fixed kk excitations
    if param.output_purity_renyi and param.method == 'ED':
        
        output_kk_purity_renyi = cavsys.compute_purity_and_Renyi_of_fixed_excitation()
        output_kk_mm_purity_renyi = cavsys.compute_purity_and_Renyi_of_fixed_k_m()
        
        #cavsys.compute_concurrence()
        
    
    # Save final distribution
    if param.output_GSdistribution and param.method == 'ED':
        out_GS_dist = cavsys.read_groundstate_distribution()
        
        #print(out_GS_dist)    
        
    if param.output_fullDistribution:
        
        if param.method in ['TWA','bbgky']:
            out_full_dist = cavsys.read_fullDistribution()
        
        if param.method == 'ED':
            out_full_dist = cavsys.read_fullstate_distribution()
    

    print("\nTime for evolving.")
    print_time(time.time()-lasttime)
    lasttime = time.time()
    
    if param.method=='ED':
        f_trace = (np.trace(cavsys.rho)).real
        f_purity = (np.trace(cavsys.rho@cavsys.rho)).real
        #f_overlap = np.trace( rho0 @ cavsys.rho ).real
        print("Trace = %g"%( f_trace ) )
        print("Final purity = %g"%( f_purity ) )
        
        #print(cavsys.project_rho_ground())
        #print("Overlap w t=0 = %g"%( f_overlap ) )
    
    
else: print("No.")





# -------------------------------------------
#           Output data
# -------------------------------------------

comments = ''
comments = comments + '# method = %s\n'%(param.method)
comments = comments + '# iterations = %i\n'%(param.iterations)
comments = comments + '# sampling = %s\n'%(param.sampling)
comments = comments + '# \n'

comments = comments + '# geometry = %s\n'%(param.geometry)
comments = comments + '# Ntotal = %i\n'%(param.Ntotal)
comments = comments + '# filling = %i\n'%(param.filling)
comments = comments + '# hilberttype = %s\n'%(param.hilberttype)
comments = comments + '# \n'

comments = comments + '# Fe = %g\n'%(float(param.Fe))
comments = comments + '# Fg = %g\n'%(float(param.Fg))
comments = comments + '# deg_e = %i\n'%(param.deg_e)
comments = comments + '# deg_g = %i\n'%(param.deg_g)
comments = comments + '# start_e = %i\n'%(param.start_e)
comments = comments + '# start_g = %i\n'%(param.start_g)
comments = comments + '# \n'

comments = comments + '# Gamma = %g\n'%(param.Gamma)
comments = comments + '# lambda0 = %g\n'%(param.lambda0)
comments = comments + '# zeeman_g = %g\n'%(param.zeeman_g)
comments = comments + '# zeeman_e = %g\n'%(param.zeeman_e)
comments = comments + '# gamma0 = %g\n'%(param.gamma0)
comments = comments + '# epsilon_ne = %g\n'%(param.epsilon_ne)
comments = comments + '# epsilon_ne2 = %g\n'%(param.epsilon_ne2)
comments = comments + '# Clebsch set to zero, q = %s\n'%( ', '.join([str(param.clebsch_zero_q[ii]) for ii in range(len(param.clebsch_zero_q))]) )
comments = comments + '# \n'

comments = comments + '# g_coh = %g\n'%(param.g_coh)
comments = comments + '# g_inc = %g\n'%(param.g_inc)
comments = comments + '# \n'

comments = comments + '# rabi_Pi = %g + %g*1j\n'%(param.rabi_Pi.real, param.rabi_Pi.imag)
comments = comments + '# rabi_Sigma = %g + %g*1j\n'%(param.rabi_Sigma.real, param.rabi_Sigma.imag)
comments = comments + '# detuning = %g\n'%(param.detuning)
comments = comments + '# \n'

comments = comments + '# Laser switch times = %s\n'%( ', '.join([str(param.switchtimes[ii]) for ii in range(len(param.switchtimes))]) )
comments = comments + '# Laser on/off = %s\n'%( ', '.join([str(param.switchlaser[ii]) for ii in range(len(param.switchlaser))]) )
comments = comments + '# \n'

comments = comments + '# IC = %s\n'%(param.cIC)
comments = comments + '# initialstate = %s\n'%('| '+' '.join(param.initialstate)+' >')
comments = comments + '# mixed_es_probabilities = %s\n'%('[ '+', '.join([ str(param.mixed_es_probabilities[ii]) for ii in range(len(param.mixed_es_probabilities)) ])+' ]')
comments = comments + '# mixed_gs_probabilities = %s\n'%('[ '+', '.join([ str(param.mixed_gs_probabilities[ii]) for ii in range(len(param.mixed_gs_probabilities)) ])+' ]')
comments = comments + '# pure_es_amplitudes = %s\n'%('[ '+', '.join([ str(param.pure_es_amplitudes[ii]) for ii in range(len(param.pure_es_amplitudes)) ])+' ]')
comments = comments + '# pure_gs_amplitudes = %s\n'%('[ '+', '.join([ str(param.pure_gs_amplitudes[ii]) for ii in range(len(param.pure_gs_amplitudes)) ])+' ]')
comments = comments + '# rotate = %s\n'%(str(param.rotate))
comments = comments + '# Omet_Pi = %g + %g*1j\n'%(param.Omet_Pi.real, param.Omet_Pi.imag)
comments = comments + '# Omet_Sigma = %g + %g*1j\n'%(param.Omet_Sigma.real, param.Omet_Sigma.imag)
comments = comments + '# \n'

comments = comments + '# phase2 = %s\n'%(str(param.phase2))
comments = comments + '# Ntstart_phase2 = %i\n'%(param.Ntstart_phase2)
comments = comments + '# Omet_Pi_phase2 = %g + %g*1j\n'%(param.Omet_Pi_phase2.real, param.Omet_Pi_phase2.imag)
comments = comments + '# Omet_Sigma_phase2 = %g + %g*1j\n'%(param.Omet_Sigma_phase2.real, param.Omet_Sigma_phase2.imag)
comments = comments + '# rabi_Pi_phase2 = %g + %g*1j\n'%(param.rabi_Pi_phase2.real, param.rabi_Pi_phase2.imag)
comments = comments + '# rabi_Sigma_phase2 = %g + %g*1j\n'%(param.rabi_Sigma_phase2.real, param.rabi_Sigma_phase2.imag)
comments = comments + '# \n'

comments = comments + '# bin_factor = %i\n'%(param.bin_factor)
comments = comments + '# which_observables = %s\n'%('[ '+', '.join([ str(param.which_observables[ii]) for ii in range(len(param.which_observables)) ])+' ]')

comments = comments + '# solver = %s\n'%(param.solver)
comments = comments + '# \n# \n'


# Initial conditions string
if param.cIC=='initialstate':
    string_IC = '_IC%s'%(''.join(param.initialstate))
if param.cIC=='mixedstate':
    string_IC = '_ICmixed'
if param.cIC=='puresupstate':
    string_IC = '_ICpuresup'
if param.cIC=='byhand':
    string_IC = '_ICbyhand'


# String to append to file names
if param.method=='ED':
    string_out = '%s_%s_Ntot%i_Ng%i_Ne%i%s%s'%(param.method, param.hilberttype, param.Ntotal, param.deg_g, param.deg_e, string_IC, param.append)
if param.method in ['MF','cumulant']:
    string_out = '%s_Ntot%i_Ng%i_Ne%i%s%s'%(param.method+param.subleading_terms, param.Ntotal, param.deg_g, param.deg_e, string_IC, param.append)
if param.method in ['TWA','bbgky']:
    string_out = '%s_Ntot%i_Ng%i_Ne%i%s_iter%i%s'%(param.method+param.subleading_terms, param.Ntotal, param.deg_g, param.deg_e, string_IC, param.iterations, param.append)


# States occupations
if param.output_occupations and param.Nt>0:
    filename = '%s/states_occ_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: time | Col >= 2: all states [g0,g1,...,e0,e1,...] | Col >=2+deg_e+deg_g: (Pi^+Pi^-) , (Sigma^+Sigma^-) , (Pi^+Sigma^- + h.c.) , (i*Pi^+Sigma^- - h.c.).\n')
        f.write('# \n# \n')
        output_data = [ [ times[tt] ] + out_occup_states[tt] for tt in range(len(times)) ]
        np.savetxt(f,output_data,fmt='%.12g')
    

# Eigenstates
if param.method=='ED' and (param.output_eigenstates or param.output_dark_renyi):
    
    if param.output_eigenstates:
        filename = '%s/eigenstates_%s.txt'%(param.outfolder, string_out)
        with open(filename,'w') as f:
            f.write(comments)
            #f.write('# Col 1: energy | Col 2: decay rate | Col 3: number of excitations | Col 4: number of states involved | Col >=5: square_abs of amplitudes of eigenstate.\n')
            f.write('# Col 1: energy | Col 2: decay rate | Col 3: number of excitations.\n')
            f.write('# \n# \n')
            floatfmt = '%.6g'
            #formatstring = '\t'.join( [floatfmt]*2 + ['%i\t%i'] + [floatfmt]*cavsys.hspace.hilbertsize )
            #output_data = [ [ cavsys.evalues[ii].real , -2*cavsys.evalues[ii].imag , cavsys.excitations_estates[ii] , cavsys.nstatesinvolved_estates[ii] ] + list(abs(cavsys.estates[:,ii])**2) for ii in range(len(cavsys.evalues)) ]
            formatstring = '\t'.join( [floatfmt]*2 + ['%i'] )
            output_data = [ [ cavsys.evalues[ii].real , -2*cavsys.evalues[ii].imag , cavsys.excitations_estates[ii] ] for ii in range(len(cavsys.evalues)) ]
            np.savetxt(f,output_data,fmt=formatstring)
        
    if param.output_dark_renyi:
        filename = '%s/darksRenyi_%s.txt'%(param.outfolder, string_out)
        with open(filename,'w') as f:
            f.write(comments)
            f.write('# Col 1: kk (excitations) | Col 2: mm (magnetic number) | Col 3: decay rate | Col 4: Renyi.\n')
            f.write('# \n# \n')
            floatfmt = '%.6g'
            formatstring = '\t'.join( ['%i'] + [floatfmt]*3 )
            np.savetxt(f, np.array(darksRenyi_data) ,fmt=formatstring)
        
        
        
# Ground-state distribution
if param.method=='ED' and param.output_GSdistribution:
    filename = '%s/GSdistrib_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1-deg_g: number of atoms in state (n_g0, n_g1,...) | Col deg_g+1: final state occupation.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( ['%i']*param.deg_g + [floatfmt] )
        np.savetxt(f,out_GS_dist,fmt=formatstring)
        
        
# Full-state distribution
if param.method in ['ED','TWA','bbgky'] and param.output_fullDistribution:
    filename = '%s/FullDistrib_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        if param.method=='ED':
            f.write('# Col 1-deg_g+deg_e: number of atoms in state (n_g0, n_g1,..., n_e0, ...) | Col deg_g+deg_e+1: final state occupation.\n')
        if param.method in ['TWA','bbgky']:
            f.write('# Col 1: population fraction of bin | 2-deg_g+deg_e+1: number of ocurrences.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        if param.method=='ED':
            formatstring = '\t'.join( ['%i']*(param.deg_g+param.deg_e) + [floatfmt] )
        if param.method in ['TWA','bbgky']:
            formatstring = '\t'.join( [floatfmt]*(len(out_full_dist[0])) )
        np.savetxt(f,out_full_dist,fmt=formatstring)


# Renyi entropy, purity, trace, overlap
if param.method=='ED' and param.output_purity_renyi and param.Nt>0:
    
    filename = '%s/purityRenyi_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: time | 2: trace | 3: purity | 4: overlap with rho(0) | 5: Renyi 1.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( [floatfmt]*5 )
        output_data = [ [ times[tt] ] + out_purity_renyi[tt] for tt in range(len(times)) ]
        np.savetxt(f, np.array(output_data) ,fmt=formatstring)
        
    filename = '%s/kkpurityRenyi_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: kk | 2: trace | 3: purity | 4: Renyi 1.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( [floatfmt]*4 )
        np.savetxt(f, np.array(output_kk_purity_renyi) ,fmt=formatstring)
        
    filename = '%s/mmkkpurityRenyi_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: kk (excitations) | 2: mm (magnetic number) | 3: trace | 4: purity | 5: Renyi 1.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( [floatfmt]*5 )
        np.savetxt(f, np.array(output_kk_mm_purity_renyi) ,fmt=formatstring)
        
        
        
# Squeezing
if param.method=='ED' and param.output_squeezing and param.Nt>0:
    
    filename = '%s/squeeJ_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: time | 2-5: squeezing inequalities for Jxyz operators.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( [floatfmt]*5 )
        output_data = [ [ times[tt] ] + out_squeezing_J[tt] for tt in range(len(times)) ]
        np.savetxt(f, np.array(output_data) ,fmt=formatstring)
        
    filename = '%s/squeeGM_%s.txt'%(param.outfolder, string_out)
    with open(filename,'w') as f:
        f.write(comments)
        f.write('# Col 1: kk | 2-d^2+1: squeezing inequalities for Gell-Mann operators.\n')
        f.write('# \n# \n')
        floatfmt = '%.6g'
        formatstring = '\t'.join( [floatfmt]*(1+cavsys.numlevels**2+1) )
        output_data = [ [ times[tt] ] + out_squeezing_GM[tt] for tt in range(len(times)) ]
        np.savetxt(f, np.array(output_data) ,fmt=formatstring)



print("\nTime for file output.")
print_time(time.time()-lasttime)
lasttime = time.time()


print("\nTotal time.")
print_time(time.time()-starttime)








"""
#### Playing with Clebsch-Gordan coefficients
cg = CG(S(3)/2, S(3)/2, S(1)/2, -S(1)/2, 1, 1)
print(cg)
cg_rat = cg.doit()
print(cg_rat)
cg_num = cg.doit().evalf()
print(cg_num)
print(S(int(2*S(3)/2))/2)

if 2*S(3)/2==3: print("yes")

print(min(S(3)/2,1.4))

for aa in range(int(2*S(5)/2)):
    print("Works")
    
fs = [ S(ff)/2 for ff in range(1,10)]
for ff in fs:
    ms = [ -ff+mm for mm in range(2*ff+1) ]
    cgs = [ CG(ff,mm,1,0,ff+1,mm).doit() for mm in ms ]
    #ms = [ -ff+mm for mm in range(2*ff) ]
    #cgs = [ CG(ff,mm,1,1,ff,mm+1).doit() for mm in ms ]
    print(cgs)
"""






