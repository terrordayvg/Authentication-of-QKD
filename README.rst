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
        
        PT_VQC is divided into 2 main parts:
        -Data generation via the User and Att codes ; Providing a set of classical output strings in the authentication.
        
        -Post-processing via DNN and static method ; Used to distinguish user from an attacker.

Contents of requirements.txt
::      

        keras==2.9.0
        matplotlib==3.5.2
        numpy==1.23.0
        pandas==1.4.3
        qiskit==0.44.2
        qiskit_aer==0.11.2
        qiskit_ibm_provider==0.5.0
        qiskit_ibmq_provider==0.19.2
        qiskit_ignis==0.7.1
        qiskit_terra==0.25.2.1
        scikit_learn==1.1.1
        scipy==1.13.1
        tensorflow==2.9.1
        torch==1.12.0+cu116
        qiskit_terra==0.22.3
        torch==1.12.0


        

Authentication of QKD: 

        An assumption is made during QKD protocols that both parties are to be trusted, what if that's not the case?
        A realistic AFC memory + noise simulation for one-way authentication of QKD is proposed in this work.
        This repository combines all the codes to produce the plots and results from the following article: arXiv:2407.03119

