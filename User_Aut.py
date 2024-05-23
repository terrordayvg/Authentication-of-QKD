

from qiskit import IBMQ, Aer, transpile, assemble
import matplotlib.pyplot as plt
from random import random
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from qiskit_ibm_provider import IBMProvider
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import random_unitary
from scipy.constants import c as clight
import random

def Gen_key(shots,user):	
	class_key=[]
	for j in range(shots):	
		random.seed(7755+j*user)
		class_key.append(random.randint(0,1))
	return class_key

import random

#Main to be iterated
def run_circ(shots,dist,j,c_key,backend,wait,T_photons,user):

    dist=dist #distance of fiber optic channel [m]
    lambd=0.17 #[db/km] 0.17[db/km] in fiber microwave frequency, 5 in above or below: in optical frequency.

    #First part of authentication circuit
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(1, 'c')
    qpe2 = QuantumCircuit(q,c)

    qpe2.h(1)
    if (c_key[j]==0):
        qpe2.cx(1,0)
    elif (c_key[j]==1):
        qpe2.cy(1,0)

	#####################################
	#Source frequency delay at generation
    T_photons=Scheduler_source(j,dist,T_photons)

	#################################### storage in memory


    qpe2=Fiber_loss(dist*pow(10,-3),lambd,qpe2,j,user) #Channel loss

    T_photons=Scheduler_memory(j,dist,T_photons,shots) # Time of storage of all qubits due to signal going in.
    T_photons=Scheduler_returnS(j,dist,T_photons,shots)# Time of storage of all qubits due to signal going out.

    T1a=T_photons[0]+wait*pow(10,-9)
    T1b=T_photons[1]+wait*pow(10,-9)

    qpe2.delay(T_photons[0]+wait*pow(10,-9), 0,unit="s") #Control - needs aditional time for photons to return from Bob's memory is needed
    qpe2.delay(T_photons[1]+wait*pow(10,-9), 1,unit="s") #Target
	#print("Time delay:")
	#print(T_photons[0]+wait*pow(10,-9),flush=True)
    qpe2=Read_loss(qpe2,T1a,T1b,j,user) #Readout loss - AFC profile
    qpe2=Fiber_loss2(dist*pow(10,-3),lambd,qpe2,j,user) #Fiber loss on the way back

	################################################################################
    #Eve changes the state in the middle with swap between qubit 2 and qubit 0
    #Initialize qubit at a random state:
	#---------------------------------
	################################
	#Verification of BoB
	#Second part of authentication circuit
    if (c_key[j]==0):
        qpe2.cx(1,0,ctrl_state=0)
    elif (c_key[j]==1):
        qpe2.cy(1,0,ctrl_state=0)
    qpe2.h(1)
    qpe2.measure(1,0)
    T_photons=Scheduler_Ver(j,dist,T_photons,shots)
		#print(qpe2)
		#Computational processing
		#sim_toro = Aer.get_backend('aer_simulator')
    t_qpe2 = transpile(qpe2, backend,seed_transpiler=j+515)
	#print(t_qpe2)
    results = execute(qpe2, backend,shots=1).result()
    finalm=results.get_counts()
    return finalm

from qiskit import QuantumCircuit, execute, BasicAer

#30 nanoseconds frequency both qubits
def Scheduler_source(j,dist,T_p):
	#Scheduler source has frequency of generation time of the circuit.

	tH=30*pow(10,-9)
	#Initialization of the photon
	T_p[0]=tH
	T_p[1]=tH
	#T_p[0]=target T_p[1]=control, [0] delay travel, [1] storage normal

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
	#print("Signal_in_storage times: Control - Target: "+str(shots-j))
	#print(T_p[j][0],T_p[j][1])

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
	#print("Signal_out_storage times: Control - Target: "+str(shots-j))
	#print(T_p[j][0],T_p[j][1])

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
def Fiber_loss2(dist,lamb,qpe2,j,user):
	#We implement this by losing the qubits
	#Measuring the state and disapearing from a memory, how do you inform Alice about that, if this information is secret
	#dist = [km]
	#lamb = [db/km]
	#p_resonant_p=1-0.06 		   #probability of resonance if happens
	p_transm=pow(10,-dist*lamb/10) # in fiber optic case 
	
	#optimal case with conversion 1588nm-637nm (assume 1538nm) - no conversion losses
	

	p_detection=0.90  #Assume fiber optic detector probability to find your photon #best parameters possible 0.9 here
	p_dark_counts=1-(1-np.exp(-0.00000025))

	p_channel=p_transm*p_detection*p_dark_counts  
	
	random.seed(662+j*user)
	s = np.random.uniform(0,1,1)
	#Measure if photon was lost
	if (s>=p_channel):
		qpe2.measure(1,0)
	return qpe2

def Fiber_loss(dist,lamb,qpe2,j,user):
	#We implement this by losing the qubits
	#Measuring the state and disapearing from a memory, how do you inform Alice about that, if this information is secret
	#dist = [km]
	#lamb = [db/km]
	#p_resonant_p=1-0.06 		   #probability of resonance if happens
	p_transm=pow(10,-dist*lamb/10) # in fiber optic case 
	
	############### detectoion
	p_detection=0.90  #Assume fiber optic detector probability to find your photon #best parameters possible 0.9 here
	p_dark_counts=1-(1-np.exp(-0.00000025))
	p_channel=p_transm*p_detection*p_dark_counts  
	
	random.seed(101414+j*user)
	s = np.random.uniform(0,1,1)
	if (s>=p_channel):
		qpe2.measure(1,0)
	return qpe2



#Quantum memory definition AFC
def Read_loss(qpe2,T1a,T1b,j,user):
	R1=0.96
	R2=0.999
	alpha=1                                      	#Alpha*L=Alpha
	lamb=1000                                       #comb FWHM
	F=40											#comb finesse
	lambd=(2*np.pi*lamb)/(np.sqrt(8*np.log(2)))
	alpha_d=((alpha)/F)*np.sqrt(np.pi/(4*np.log(2)))

	t=T1a #time of storage for memory of alice
	nu=(4*(alpha_d)*(alpha_d)*np.exp(-2*(alpha_d))*(1-R1)*(1-R1)*R2*np.exp(-t*t*lambd*lambd))/pow((1-np.sqrt(R1*R2)*np.exp(-alpha_d)),4)


	t2=T1b #time of storage for memory of bob
	nu2=(4*(alpha_d)*(alpha_d)*np.exp(-2*(alpha_d))*(1-R1)*(1-R1)*R2*np.exp(-t2*t2*lambd*lambd))/pow((1-np.sqrt(R1*R2)*np.exp(-alpha_d)),4)
	

	p_channel=nu 
	p_channel2=nu2 

	#Photon is lost due to storage efficiency? Yes or no for both memories
	random.seed(1041216+j*user)
	s = np.random.uniform(0,1,1)    
	if (s>=p_channel):
		qpe2.measure(1,0)

	random.seed(101566+j*user)
	s2 = np.random.uniform(0,1,1)
	if (s2>=p_channel2):
		qpe2.measure(0,0)  

	return qpe2

	#print(p_transm)

def task_wrapper(args):
    # call task() and unpack the arguments
    return run_circ(*args)

from qiskit.test.mock import FakeVigo
from qiskit.providers.fake_provider import FakeManilaV2


def CX_BB84_noisy(cores):
	#Substitute whats inside save account to your IBM profile
	IBMQ.save_account(" ",overwrite=True) 
	provider = IBMQ.load_account()
	backend_lima = provider.get_backend('ibm_brisbane')
	noise_model= NoiseModel.from_backend(backend_lima)
	coupling_map = backend_lima.configuration().coupling_map
	basis_gates = noise_model.basis_gates
	backend = AerSimulator(noise_model=noise_model,
                       coupling_map=coupling_map,
                       basis_gates=basis_gates) 

	Vec=[]
	Store_V=[]
	###################### example: 4 km of distance between Alice and Bob , 1mus storage time of AFC, 500 attackers (statistics), lambda=300 (shots)  
	V_count=0
	nusers=500
	maxshots=301
	wait=1000
	dist=4000
	for shots in range(300,maxshots,20): #shots increase
		for user in range(1,nusers+1,1):
			T_photons=[[0,0]]*shots
			c_key=Gen_key(shots,user)
			args = [[shots,dist,i,c_key,backend,wait,T_photons[i],user] for i in range(shots)]
			V_count=0
			with Pool(cores) as pool:
				finalm=pool.map(task_wrapper, args)
				arx=[]
				for j in range(shots):
					for a in finalm[j].keys():
						arx.append(a)
						if int(a)==0:
							V_count=V_count+1
			#print(arx)
			Vec.append(V_count/shots)
			Store_V.append(arx)
		print(" ")
		print("shots :"+str(shots))
		print(Store_V)

	return Vec,Store_V,shots

def plot_func(V,Store,shots):
	fig = plt.figure(figsize=(10,10))
	axa = fig.add_subplot(111)
	#rect = [0.35,0.1,0.6,0.6]
	axa.plot(Store,V,"--", color="red",label='CX-BB84')
	axa.set_title("CX-BB84 in realistic-sim")
	axa.set_xlabel("Distance [m]")
	axa.set_ylabel("Normalized counts")
	axa.grid(which='major')
	axa.grid(which='minor')
	#axa.plot(X2,Y2,"--", color="black",label='Google_D3')
	axa.legend(loc ="upper left")
	plt.show()

import qiskit
if __name__ == '__main__':
	#Paralellization, choose amount of cores for multiprocessing
	cores=1
	print(qiskit.__qiskit_version__)
	V,Store,shots=CX_BB84_noisy(cores)
	print("Attacker data:") 
	print(V)
	print("-")
	print(Store)
	print("-")
	print(shots)
	#plot_func(V,Store,shots)
	with open('Att_data_authentication.txt', 'w') as output:

		output.write("\n")

		output.write("User"+str(shots)+": \n")
		output.write(str(Store))
    	
    
	output.close()
