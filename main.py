### Code created by: Vladlen Galetsky - 10/10/2024 #########################################################################
### Purpose: Simulating realistically the authentication protocol in :arXiv:2407.03119 #####################################
### Combines perceval (optical channel definition) + qutip (mid operations) + qiskit (decoherence + dephasing channel) #####
############################################################################################################################
##
## ------ CX / CZX optically post-processed -- Fiber optic channel -- Q mem channel -- Fiber optic channel -- Authentication 
## Returns: Probability of correct authentication: "Vec" + string of measurements [0,1,0,0,0...] : "finalm"
##

# Imports : numpy, matplotlib, perceval, qiskit
import sys
import numpy as np
import sdeint
import matplotlib
import matplotlib.pyplot as plt
import perceval.components.unitary_components as comp
import qiskit
import perceval
import perceval as pcvl
import sys
from perceval.converters import QiskitConverter
from perceval.components import catalog
from qiskit.quantum_info import DensityMatrix
from perceval.backends import SLOSBackend
from perceval.simulators import Simulator
from perceval.algorithm import Analyzer, Sampler
from perceval.components import BS, PERM, Port
from perceval.utils import Encoding
from perceval.components.linear_circuit import Circuit
from perceval.backends import NaiveBackend
from perceval.utils.postselect import PostSelect
from perceval.components import catalog, Processor, BS
from perceval.components.source import Source
from perceval.algorithm import ProcessTomography as PT
from perceval.algorithm.tomography import is_physical, process_fidelity
from perceval.algorithm import ProcessTomographyMLE, StateTomographyMLE
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from qiskit import transpile, assemble
from scipy.constants import c as clight
import random
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Chi, Choi, Statevector
from qiskit_experiments.library.tomography import ProcessTomography
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace 
from qiskit_aer.library import save_density_matrix
import qutip as qp
from qutip.core.states import basis
from qutip.measurement import measure
from qutip import Qobj
from qiskit.quantum_info import Chi, Choi, Statevector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import partial_trace 

# Initialization in this simulation is done between CX and CY instead of CZX (which is equivalent in terms of noise).
def qiskit_converterl():

	#Circuit 1 for cx (qc)
	qc = qiskit.QuantumCircuit(2)
	qc.h(1)
	qc.cx(1, 0)

	#Decomposed circuit for cy (qc2)
	qc2 = qiskit.QuantumCircuit(2)
	qc2.h(1)
	qc2.sdg(0)
	qc2.cx(1,0)
	qc2.s(0)

	#Using processor- perceval conversion cx -----------------------------------------------------------------
	qiskit_converter = QiskitConverter(catalog, backend_name="Naive")
	quantum_processor = qiskit_converter.convert(qc, use_postselection=True)

	#Using processor- perceval conversion czx -----------------------------------------------------------------
	qiskit_converter2 = QiskitConverter(catalog, backend_name="Naive")
	quantum_processor2 = qiskit_converter2.convert(qc2, use_postselection=True)

	
	qpt = PT(operator_processor=quantum_processor)		# Process tomography from perceval processor cx
	chi_op = qpt.chi_matrix()  							# computing the chi matrix

	qpt2 = PT(operator_processor=quantum_processor2)	# Process tomography from perceval processor czx
	chi_op2 = qpt2.chi_matrix()  						# computing the chi matrix

	#Chi operator matrix to chi operator class in qiskit------------------------------------------------
	chi_opt = Chi(4*chi_op)                             # 4 is the normalization correction from 1/2^N, with N being the qubit dimension
	chi_opt2 = Chi(4*chi_op2)

	Init_state=[1,0,0,0]

	#Evolution of a state vector |00> to be evolved by chi matrix
	rho_p1=chi_opt._evolve(Init_state)
	rho_p2=chi_opt2._evolve(Init_state)

	return rho_p1,rho_p2


#Authentication of the qubits
def qiskit_converter2t(rho,c_key):

	#Circuit 1 for cx
	qc = qiskit.QuantumCircuit(2)
	qc.cx(1, 0)
	qc.h(1)

	#Decomposed circuit 1 for cy 
	qc2 = qiskit.QuantumCircuit(2)
	qc2.sdg(0)
	qc2.cx(1,0)
	qc2.s(0)
	qc2.h(1)

	#Using processor------------------------------------------------------------------
	qiskit_convertera = QiskitConverter(catalog, backend_name="Naive")
	quantum_processor = qiskit_convertera.convert(qc, use_postselection=True)

	qiskit_converter2a = QiskitConverter(catalog, backend_name="Naive")
	quantum_processor2 = qiskit_converter2a.convert(qc2, use_postselection=True)


	#Process tomography on the processor to understand its chi_matrix
	qpt = PT(operator_processor=quantum_processor)
	chi_op = qpt.chi_matrix()  # computing the chi matrix

	qpt2 = PT(operator_processor=quantum_processor2)
	chi_op2 = qpt2.chi_matrix()  # computing the chi matrix 
	
	#chi operator matrix to chi operator class in qiskit - Qiskit
	chi_opt = Chi(4*chi_op)
	chi_opt2 = Chi(4*chi_op2) #4 is renormalization 1/2^n from perceval to qiskit

	rhoM=[]
	#CX option
	if (c_key==0):
		rhoM.append(chi_opt._evolve(rho))

	#CY option
	elif (c_key==1):
		rhoM.append(chi_opt2._evolve(rho))
	
	#Returns evolved density matrix before measurement
	return rhoM[0]


#Qiskit process tomography of the transmission channel + Qmemory channel
def Ptom(qc,backend):
	qpt = ProcessTomography(qc)
	qptdata1 = qpt.run(backend, shots=(1000)).block_for_results()
	#Result
	result=qptdata1.analysis_results("state").value #Obtain choi matrix outside from the experiment with backend, decoherence and dephasing
	return result


#Measure q1
def MeasureF(rho):
	#Define the basis of measurement I otimes Z and Z otimes I (PZ1 and PZ2 respectively)
	Z0, Z1 = qp.ket2dm(basis(2, 0)), qp.ket2dm(basis(2, 1))
	PZ1 = [qp.tensor(Z0, qp.identity(2)), qp.tensor(Z1, qp.identity(2))]
	PZ2 = [qp.tensor(qp.identity(2), Z0), qp.tensor(qp.identity(2), Z1)]


	a=Qobj(rho,dims=[[2,2],[2,2]]) 	  			#qutip object of density matrix
	qa=measure(a, PZ2)                			#output: [meas, obj rho]

	return qa[0]

#Conditional measurement depending on the M output, M=[0,0] (no measurement), M=[0,1] (q1 measured), M=[1,0] (q0 measured), M=[1,1] (both qubit measured)
def CMeasure(rho,M):

	#Define the basis of measurement I otimes Z and Z otimes I (PZ1 and PZ2 respectively)
	Z0, Z1 = qp.ket2dm(basis(2, 0)), qp.ket2dm(basis(2, 1))
	PZ1 = [qp.tensor(Z0, qp.identity(2)), qp.tensor(Z1, qp.identity(2))]
	PZ2 = [qp.tensor(qp.identity(2), Z0), qp.tensor(qp.identity(2), Z1)]
	qtv=[]
	qtv2=[]
	probv=[]

	#If M=[1,0] or M=[1,1]
	if (M[0]==1):
		a=Qobj(rho,dims=[[2,2],[2,2]]) 	  			  #qutip
		qa=measure(a, PZ1)                			  #output: [meas, obj rho]
		qa1=DensityMatrix(np.array(qa[1].full())) 	  #qiskit obj
		#If M=[1,1]
		if (M[1]==1):
			qa=measure(qa1, PZ2)                	  #qutip: [meas, obj rho]
			qa1=DensityMatrix(np.array(qa[1].full())) #qiskit obj
		
		return qa1

	#If M=[0,1]
	elif (M[1]==1):
		a=Qobj(rho,dims=[[2,2],[2,2]]) 
		qa=measure(a, PZ2)     
		qa1=DensityMatrix(np.array(qa[1].full()))
		return qa1

	return rho

#Generation of the classical key G_key=[0,1,0,1,1...] means [CX,CY,CX,CY,CY...]
def Gen_key(shots,user):	
	class_key=[]
	for j in range(shots):	
		random.seed(7755+j*user)
		#Choice between cx or czx initialization: classical key generation
		class_key.append(random.randint(0,1))

	return class_key


#Main to be iterated
def run_circ(shots,dist,j,c_key,backend,wait,T_photons,user,rho1,rho2,attack):

	#Randomness in the seed of the efficiency model
	np.random.seed(j)

	M=[0,0]    
	dist=dist  # distance of fiber optic channel [m]
	lambd=0.17 # [db/km] 0.17[db/km] in fiber microwave frequency, 5 in above or below: in optical frequency.

    # Definition of noise model for decoherence and dephasing in qiskit
	q = QuantumRegister(2, 'q')
	c = ClassicalRegister(1, 'c')
	qpe2 = QuantumCircuit(q,c)
   
 
	#Source frequency delay at generation
	T_photons=Scheduler_source(j,dist,T_photons)

	#Storage in memory
	M=Fiber_loss(dist*pow(10,-3),lambd,j,user,M,c_key) #Channel loss

	T_photons=Scheduler_memory(j,dist,T_photons,shots) # Time of storage of all qubits due to signal going in.
	T_photons=Scheduler_returnS(j,dist,T_photons,shots)# Time of storage of all qubits due to signal going out.

	T1a=T_photons[0]+wait*pow(10,-9)
	T1b=T_photons[1]+wait*pow(10,-9)

	qpe2.delay(T_photons[0]+wait*pow(10,-9), 0,unit="s") #Control - needs aditional time for photons to return from Bob's memory is needed
	qpe2.delay(T_photons[1]+wait*pow(10,-9), 1,unit="s") #Target

	M=Read_loss(T1a,T1b,j,user,M,c_key) 				#Readout loss - AFC profile
	M=Fiber_loss2(dist*pow(10,-3),lambd,j,user,M,c_key) #Fiber loss on the way back


	#Process tomography (in qiskit enviroment)
	#Obtain Chi matrix from [fiber+Mem+fiber] channel
	result=Ptom(qpe2,backend)
	rho_post=[]
	#CX option
	if (int(c_key)==0):
		rho_post.append(result._evolve(rho1))

	#CY option
	elif (int(c_key)==1):
		rho_post.append(result._evolve(rho2))

	#Attacker is present, attack flag=0
	if (attack==1):
		M[0]=1

	rho=CMeasure(rho_post[0],M) #Measurement at the end of the [fiber+Mem+fiber] depending on noise (efficiencies)

	
	#Second part of the optical simulator (authentication)
	rho=qiskit_converter2t(rho,c_key)
	#Measurement of q1 to verify if its 0 or 1
	#returns output of measurement in Pauli Z basis
	outq0=MeasureF(rho)

	return outq0


#Scheduler source has frequency of generation time of the circuit.
def Scheduler_source(j,dist,T_p):

	tH=30*pow(10,-9) #30 nanoseconds frequency both qubits
	#Initialization of the photon
	T_p[0]=tH
	T_p[1]=tH

	#delay time
	return  T_p

#Time for storage of states (each photon awaits for the other photon to arrive)
def Scheduler_memory(j,dist,T_p,shots):
	#Fiber optic delay time of photon depending on the distance.
	c=clight
	n=1.44 #refractive index in glass
	vel=c/n 
	ttrav=dist/vel #photon delay time

	#Position in memory in terms of storage times
	A=30*pow(10,-9)          #interval betweeen sending (frequency of the source 30nanoseconds)
	tpul_store=30*pow(10,-9) #storage waiting time to input states

	T_p[0]=(j)*(A+tpul_store)+ttrav #Time of storage during the control
	T_p[1]=(j)*(A+tpul_store) #Time of storage during the target


	return T_p

#Additional photon time to wait to be sent back in storage
def Scheduler_returnS(j,dist,T_p,shots):
	c=clight #speed of light in vacum
	n=1.44 #refractive index in glass
	vel=c/n 
	ttrav=dist/vel

	tpul_read=30*pow(10,-9) #recover time 60ns
	T_p[0]=(shots-j-1)*(tpul_read)+ttrav+T_p[0] #Control
	T_p[1]=(shots-j-1)*(tpul_read)+T_p[1] #Target


	return T_p

#Time to execute second part of authentication circuit - Hadamard + CX
def Scheduler_Ver(j,dist,T_p,shots):

	tH=30*pow(10,-9)
	tCX=30*pow(10,-9)

	#Initialization of the photon
	T_p[0]=T_p[0]+tH+tCX
	T_p[1]=T_p[0]+tH+tCX


	return T_p


#Fiber loss of dark counts, transmission loss, detection loss (way back)
def Fiber_loss2(dist,lamb,j,user,M,c_k):
	#We implement this by losing the qubits (measuring them)
	#dist = [km]
	#lamb = [db/km]
	p_transm=pow(10,-dist*lamb/10) # eff fiber optic case 
								   #optimal case with conversion 1588nm-637nm (assume 1538nm) - no conversion losses
	

	p_detection=0.95  			   			#Assume fiber optic detector probability to find your photon 
	p_dark_counts=1-(1-np.exp(-0.00000025)) #dark count eff

	p_channel=p_transm*p_detection*p_dark_counts  #Comulative probability for the channel of losing the photon
	#CX option
	if (c_k==0):
		random.seed(1216+j*user)
		s = np.random.uniform(0,1,1)    
		if (s>=p_channel):
			M[0]=1
	#CY option
	elif(c_k==1):
		random.seed(1041216+j*user)
		s = np.random.uniform(0,1,1)    
		if (s>=p_channel):
			M[1]=1

	return M

def Fiber_loss(dist,lamb,j,user,M,c_k):
	#We implement this by losing the qubits (measurement of qubits)
	#dist = [km]
	#lamb = [db/km]
	p_transm=pow(10,-dist*lamb/10) # fiber optic eff
	
	############### detectoion
	p_detection=0.95  #Assume fiber optic detector probability to find your photon 
	p_dark_counts=1-(1-np.exp(-0.00000025))
	p_channel=p_transm*p_detection*p_dark_counts  
	random.seed(101414+j*user)

	#CX option
	if (c_k==0):
		random.seed(1216+j*user)
		s = np.random.uniform(0,1,1)    
		if (s>=p_channel):
			M[0]=1

	#CY option
	elif(c_k==1):
		random.seed(1041216+j*user)
		s = np.random.uniform(0,1,1)    
		if (s>=p_channel):
			M[1]=1
			
	return M



#Quantum memory definition AFC
def Read_loss(T1a,T1b,j,user,M,c_k):
	R1=0.96											#Reflectivity parameters R1 and R2
	R2=0.999
	alpha=1                                      	#Alpha*L=Alpha
	lamb=1000                                       #comb FWHM
	F=40											#comb finesse
	lambd=(2*np.pi*lamb)/(np.sqrt(8*np.log(2)))
	alpha_d=((alpha)/F)*np.sqrt(np.pi/(4*np.log(2)))

	t=T1a #time of storage for memory of alice
	nu=(4*(alpha_d)*(alpha_d)*np.exp(-2*(alpha_d))*(1-R1)*(1-R1)*R2*np.exp(-t*t*lambd*lambd))/pow((1-np.sqrt(R1*R2)*np.exp(-alpha_d)),4)    #Eff of AFC Mem 1


	t2=T1b #time of storage for memory of bob
	nu2=(4*(alpha_d)*(alpha_d)*np.exp(-2*(alpha_d))*(1-R1)*(1-R1)*R2*np.exp(-t2*t2*lambd*lambd))/pow((1-np.sqrt(R1*R2)*np.exp(-alpha_d)),4) #Eff of AFC Mem 2
	

	p_channel=nu 
	p_channel2=nu2 


	#Photon is lost due to storage efficiency? Yes or no for both memories
	#Memory Alice
	random.seed(1216+j*user)
	s = np.random.uniform(0,1,1)    
	if (s>=p_channel):
		M[0]=1

	#Memory Bob
	random.seed(155+j*user)
	s2 = np.random.uniform(0,1,1)
	if (s2>=p_channel2):
		M[1]=1
	
	return M

#Initialization of parameters
def init_p(argv):
	#Restrictions:----------------------------
	#default 1, ------, [m], [s], default 1, default 1
	#Parameters:------------------------------
	#minshots,maxshots,dist,wait,nusers, cores, attacker present (0 or 1)
	#argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]
	#-----------------------------------------

	#Initialize parameters
	Val=[1,1000,1000,1,1,0]
	Lpar=["shots","dist","wait","users","cores","attack"] 						 
	for i in range(1,len(argv),1):
		res = ''.join([j for j in argv[i] if not j.isdigit()])
		digit = ''.join([j for j in argv[i] if j.isdigit()])

		for j in range(len(Lpar)):
			if(res==Lpar[j]):
				Val[j]=int(digit)  #every float in input approximated to int

	
	#Error-protection
	if (Val[3]<1):
		raise ValueError('Number of user must be >1')

	if (Val[4]<1):
		raise ValueError('Number of cores must be >1')

	if (Val[5]!=0) and (Val[5]!=1):
		raise ValueError('Invalid flag for attacker: attack0 = no man in the middle, attack1= man in the middle')

	print("Parameters:")
	print("shots, dist, wait, users, cores, attacker")
	print(Val[0],Val[1],Val[2],Val[3],Val[4], Val[5])

	return Val[0],Val[1],Val[2],Val[3],Val[4],Val[5]

#Multi-processing 
def task_wrapper(args):
    return run_circ(*args)

#Saves raw output
def save_file(Vec,attack):
	
	with open('Authen_'+str(attack)+'.txt', 'w') as output:
		output.write("\n")
		output.write(str(Vec))
    
	output.close()

#Main process
def CX_BB84_noisy(cores,rho1,rho2,backend,shots,dist,wait,nusers,attack):
	Vec=[]
	Store_V=[]
	V_count=0
	print("Starting authentication processing...")
	for user in range(1,nusers+1,1):
		T_photons=[[0,0]]*shots
		c_key=Gen_key(shots,user)
		args = [[shots,dist,i,c_key[i],backend,wait,T_photons[i],user,rho1,rho2,attack] for i in range(shots)]
		V_count=0
			#Multi-processing of the size of the cores
		with Pool(cores) as pool:
			finalm=pool.map(task_wrapper, args)
			pool.close()
			pool.join()
	
			print("Output vector: -----------------------------------------------------------",flush=True)
			print(finalm,flush=True)

			Store_V.append(finalm)
			for l in range(len(finalm)):
				if (finalm[l]==0):
					V_count=V_count+1

			
			Vec.append((V_count)/shots)
			print("Prob of correct authent vector-------:", flush=True)
			print(Vec,flush=True) 			#Probability of correct authentication for each trials
			#Save raw data
		save_file(Store_V,attack)

    	
    
	return Vec,Store_V



if __name__ == "__main__":

	#Initialize input parameters
	shots,dist,wait,nusers,cores,attack=init_p(sys.argv)

	#Input your qiskit token so you can obtain the ibm_sherbrooke backend
	token=" "
	#QiskitRuntimeService.save_account(channel="ibm_quantum", token=token) #If it is your first time running, to save the token use this line of code
	service = QiskitRuntimeService(channel="ibm_quantum")
	backend = service.backend("ibm_sherbrooke")                            #Backend used to obtain the T1 and T2 pamaters

	noise_model= NoiseModel.from_backend(backend)                          #In the aer simulator import the noise parameters from the backend
	coupling_map = backend.configuration().coupling_map
	basis_gates = noise_model.basis_gates
	backend = AerSimulator(noise_model=noise_model,
                       coupling_map=coupling_map,
                       basis_gates=basis_gates )

	print("Starting initialization...")
	rho1,rho2=qiskit_converterl()										   #rho1 (cx initialization) and rho2 (czx initialization) in an optical circuit

	CX_BB84_noisy(cores,rho1,rho2,backend,shots,dist,wait,nusers,attack)   #Main processing