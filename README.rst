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

Architecture
-----

.. image:: /Img/arc.png
  :alt: Architecture of CX/CZX authentication protocol used in the simulation.

One directional CX/CZX BB84 embedded authentication, QM defines the Atomic frequency comb cavity enhanced memory which stores the states at the server-user end. 




Usage
-----
You either run the code independently (main.py and DNN_binary_class.py) or sequentially using snakemake by running (here --cores x means the amount of cores for multiple file processing to create the attacker and user dataset (max=2) from main.py):

::
        snakemake --cores 1

::::
        
        The authentication protocol code is divided into two parts:
        
        -main.py : Generates the simulation with CPU multiprocessing;
        Input: * `distance:` Between Alice and Bob [m].
               * `time:` Time to retain qubits in a quantum memory [s].
               * `cores:` Amount of cores used in multiprocessing .
               * `nusers:` Amount of users - repeats the trials for each user, the simulator iterates from (1,nusers,step).
               * `maxshots:` λ, the simulator iterates from (1,maxshots,step).

        Output: * `Vec:` probability vector of correct authentication.
                * `Store_V:` Output vector of measurements for authentication.

        Additionally: To use main.py the perceval/components/unitary_components.py will be modified.
                      To use main.py the qutip/measurement.py will be modified.

        
        -DNN_binary_class.py: Generates the weights for the binary classification for the input data.
        
        Input:  * `At:` Output vector of measurements for authentication for Attacker.
                * `E:` Output vector of measurements for authentication for User.
        
        Output: * `Roc curve plot`
                * `Accuracy, cross entropy plot`
                * `Att:` Probability of correctly predicted authentication
                
Contents of requirements.txt
::      

        keras==2.9.0
        matplotlib==3.5.2
        numpy==2.1.2
        pandas==1.4.3
        perceval_quandela==0.11.1
        qiskit==1.2.4
        qiskit_aer==0.15.1
        qiskit_experiments==0.7.0
        qiskit_ibm_runtime==0.30.0
        qiskit_ibmq_provider==0.19.2
        qiskit_ignis==0.7.1        
        qiskit_terra==0.25.2.1
        qutip==5.0.4
        scikit_learn==1.1.1
        scipy==1.14.1
        sdeint==0.3.0
        tensorflow==2.9.0
        tensorflow_intel==2.16.1
        torch==1.12.0+cu116
        qiskit_terra==0.22.3
        torch==1.12.0
        snakemake==7.32.4


        

Authentication of QKD: 

        An assumption is made during QKD protocols that both parties are to be trusted, what if that's not the case?
        A realistic AFC memory + noise simulation for one-way authentication of QKD is proposed in this work.
        This repository combines all the codes to produce the plots and results from the following article: arXiv:2407.03119

