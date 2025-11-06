
#Ignore deprecation warnings for old packages - must be changed if you plan to update to a newer version of perceval or qiskit


import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys, os
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator             #used to substitute connection with IBM API, to test offline
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit.quantum_info import Choi, partial_trace, Operator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main


#Auxiliary function to create density matrices from state initialization ------
def dm(a, b):
    rho = DensityMatrix.from_label(str(a)+str(b))
    return rho

# ---------------------------
# Parametrize simplifies repetitive checks for measurement function
# ---------------------------
@pytest.mark.parametrize(
    "state, M, expected",
    [
        ((0,1), [0,0], (0,1)),  # no measurement
        ((0,1), [0,1], (0,1)),  # measure 2nd qubit
        ((1,0), [1,0], (1,0))   # measure 1st qubit
    ]
)
# ---------------------------
# Unit testing - Measurement Function - identity
# ---------------------------
def test_CMeasure(state, M, expected):
    a, b = state
    rho = dm(a, b)                           # numpy density matrix in
    out = main.CMeasure(rho, M)              # mixed-type output

    expected_dm = dm(*expected)

    assert np.allclose(out, expected_dm)

#Measure q1
@pytest.mark.parametrize(
    "state, expected",
    [
        ((0,0), 0),  # no measurement
        ((0,1), 1)   # measure 2nd qubit
    ]
)

def test_MeasureF(state, expected):
    a, b = state
    rho = dm(a, b)           # prepare |ab><ab| density matrix
    out = main.MeasureF(rho) # returns 0 or 1

    assert out == expected


#Test process tomography function for identity and verify its validity if it is a CPTP map (complete positive trace preserving map)
def test_Ptom():
    qc = QuantumCircuit(2)
    ideal_backend = AerSimulator()
    result=main.Ptom(qc,ideal_backend)

    eigvals = np.linalg.eigvalsh(result.data)

    J_mat = np.round(result.data,1)  # rounds to 1 decimal point, this this due to low precision of the process tomography 1000shots only in the simulation to be faster
    # Reshape to (input_dim, output_dim, input_dim, output_dim)
    J_reshaped = J_mat.reshape(4, 4, 4, 4)

    # Partial trace over output (second index)
    pt = np.einsum('ijik->jk', J_reshaped)


    assert np.allclose(result.data, result.data.conj().T) #Hermitian check 
    assert np.all(eigvals >= -1e-12)  #eigenvalues are positive up to a floating point
    assert np.allclose(pt, np.eye(4)) #is it trace preserving

    #So its a CPTP map

# ---------------------------
# Mock run_circ to isolate CX_BB84_noisy logic
# ---------------------------
def test_run_circ():
    """Test algorithm loop logic without real quantum calls"""

    ideal_backend = AerSimulator()
    psi = Statevector.from_label('00')
    rho1 = DensityMatrix(psi)
    rho2=DensityMatrix(psi)

    out_p= main.run_circ(
        shots=1,dist=0,j=0,c_key=0,backend=ideal_backend,wait=0,T_photons=[0,0],user=1,rho1=rho1,rho2=rho2,attack=0
    )

    assert isinstance(out_p,int

    ) #expected to get 0, after measurement
   
