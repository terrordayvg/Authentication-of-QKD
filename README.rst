=========================================================================================================================
Entanglement-assisted authenticated BB84 protocol
=========================================================================================================================


.. image:: https://dl.circleci.com/status-badge/img/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup.svg?style=shield&circle-token=41de148cb83684dd3c53509e74c3048071434118
        :target: https://dl.circleci.com/status-badge/redirect/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
        :target: https://www.python.org/downloads/release/python-3110/


Installation of required libraries
-----

::

    install -r requirements.txt


Necessary library modifications!
-----

In the folder **"Modified_libraries"** are the updated files for perceval and qutip (changes explained in Modified_libraries/comment.txt). In source code you can find the original files at:
    * perceval/components/unitary_components.py (adds the loss of beam splitters and phase shifters according to: arXiv:2311.10613v3)
    * qutip/measurements.py (creates extra protection with the choice of probabilities during Z-basis measurement)

Update them by creating a development directory, see more at: https://stackoverflow.com/questions/23075397/python-how-to-edit-an-installed-package

Architecture
-----

.. image:: /Img/Arc_up.png
  :alt: Architecture of CX/CZX authentication protocol used in the simulation.

One directional CX/CZX BB84 embedded authentication, QM defines the Atomic frequency comb cavity enhanced memory which stores the states at the server-user end. In the output measurement (Z-basis) Alice verifies Bob by measuring [0,1,1,0..]. For noiseless authentication 0s would correspond to CX and 1s to CZX.




Usage
-----
::
        
IBM Token:
        * You should in **main.py** update **token=' '** with your IBM token API to use the IBM backend for the decoherence and dephasing parameters. 

You either run the code independently (main.py and DNN_binary_class.py) or sequentially using snakemake by running ( "--cores x" means the amount of cores for multiple file processing to create the attacker and user dataset from main.py):

Command example:
::
        snakemake --cores 1


The authentication protocol code is divided into two parts:


        ::

main.py
        
                Generates the simulation with CPU multiprocessing for user (attack0) or attacker (attack1);

        Command example: (10 shots, distance 1km, wait 1000ns, amount of authentications 1, cores in multiprocessing of shots 1, user is using the protocol) 
        ::
                python main.py shots10 dist1000 wait1000 users1 cores1 attack0

        Input: 
               * `dist:` Between Alice and Bob [m].
               * `wait:` Time to retain qubits in a quantum memory [ns].
               * `cores:` Amount of cores used in multiprocessing for the simulation (its different from the file multiprocessing present in snakemake --cores x).
               * `users:` Amount of users - repeats the trials for each user, the simulator iterates from (1,nusers,step).
               * `shots:` λ, lenght of the authentication string.
               * `attack:` 0 (user authenticating) or 1 (attacker authenticating).

        Output: 
                * `Vec:` probability vector of correct authentication.
                * `Store_V:` Output vector of measurements for authentication.
                * `Authen_x.txt:` Output file with the Store_V in it

        

DNN_binary_class.py

                Generates the DNN binary classification for the input data;
        
        Input:  
                * `At:` Output vector of measurements for authentication for Attacker.
                * `E:` Output vector of measurements for authentication for User.

                        or

                * `Authen_1.txt:` Output file from main.py with vector of measurements for authentication for Attacker.
                * `Authen_0.txt:` Output file from main.py with vector of measurements for authentication for User.
                
        
        Output: 
                * `Roc curve plot`.
                * `Accuracy, cross entropy plot`.
                * `Att:` Probability of correctly predicted authentication.
                * `DataDNN.txt: Att data in a file`.

                
Contents of requirements.txt
-----

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


        

Authentication of QKD background:
-----

        An assumption is made during QKD protocols that both parties are to be trusted, what if that's not the case?
        A realistic AFC memory + noise simulation for one-way authentication of QKD is proposed in this work.
        This repository combines all the codes to produce the plots and results from the following article: arXiv:2407.03119.

        Please cite it if the code is used.


