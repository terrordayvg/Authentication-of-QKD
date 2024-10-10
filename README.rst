=========================================================================================================================
``Entanglement-assisted authenticated BB84 protocol``
=========================================================================================================================


.. image:: https://dl.circleci.com/status-badge/img/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup.svg?style=shield&circle-token=41de148cb83684dd3c53509e74c3048071434118
        :target: https://dl.circleci.com/status-badge/redirect/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
        :target: https://www.python.org/downloads/release/python-3110/


Installation of required libraries

::

    install -r requirements.txt

Usage
-----

::

    Usage:
        
        The authentication protocol code is divided into two parts:
        
        -main.py : Generates the simulation with CPU multiprocessing;
        Input: distance: Between Alice and Bob [m].
               time: Time to retain qubits in a quantum memory [s].
               cores: Amount of cores used in multiprocessing .
               nusers: Amount of users - repeats the trials for each user, the simulator iterates from (1,nusers,step).
               maxshots: λ, the simulator iterates from (1,maxshots,step).

        Output: Vec: probability vector of correct authentication.
                Store_V: Output vector of measurements for authentication.

        
        -Post-processing via DNN and static method ; Used to distinguish user from an attacker.

Contents of requirements.txt
::      

        keras==2.9.0
        pandas==1.4.3
        scikit_learn==1.1.1
        scipy==1.13.1
        tensorflow==2.9.1
        torch==1.12.0+cu116
        torch==1.12.0
        matplotlib==3.5.2
        numpy==2.1.2
        perceval_quandela==0.11.1
        qiskit==1.2.4
        qiskit_aer==0.15.1
        qiskit_experiments==0.7.0
        qiskit_ibm_runtime==0.30.0
        qiskit_ibmq_provider==0.19.2
        qiskit_ignis==0.7.1
        qiskit_terra==0.25.2.1
        qutip==5.0.4
        scipy==1.14.1
        sdeint==0.3.0
        qiskit_terra==0.22.3


        

Authentication of QKD: 

        An assumption is made during QKD protocols that both parties are to be trusted, what if that's not the case?
        A realistic AFC memory + noise simulation for one-way authentication of QKD is proposed in this work.
        This repository combines all the codes to produce the plots and results from the following article: arXiv:2407.03119

