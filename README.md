=========================================================================================================================
``Non-Unitary Channel QPUFs``
=========================================================================================================================


.. image:: https://dl.circleci.com/status-badge/img/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup.svg?style=shield&circle-token=41de148cb83684dd3c53509e74c3048071434118
        :target: https://dl.circleci.com/status-badge/redirect/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup



.. image:: https://codecov.io/gh/terrordayvg/PT_VQC-Tomography/graph/badge.svg?token=880RTY0T96
        :target: https://codecov.io/gh/terrordayvg/PT_VQC-Tomography

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
        :target: https://www.python.org/downloads/release/python-3110/


Installation of required libraries

::

    install -r requirements.txt

Usage
-----

::

    Usage:
        Puf_sim is divided into 5 distinct folders, responsible for:
        - PE-QPUF
        - RCNOT-QPUF
        - Quantum_Money
        - Physical implementation
        - Img

    In each folder:
        - A Comment.txt file explains the files hierarchy and objective with input and output functions

    Tests:
        - CircleCI covers up the continuous integration of the code and coverage of the basic functions (.circleci folder).

Contents of requirements.txt

::

        matplotlib==3.5.2
        numpy==1.23.0
        pandas==1.4.3
        PennyLane==0.31.0
        qiskit==0.39.4
        qiskit_aer==0.11.2
        qiskit_ibmq_provider==0.19.2
        qiskit_ignis==0.7.1
        qiskit_terra==0.22.3
        qutip==4.7.1
        scikit_learn==1.1.1
        scipy==1.8.1
        sympy==1.12
        torch==1.12.0
        tqdm==4.64.1
        pytest==7.4.2


What is a Non-Unitary Channel QPUF?
A quantum physical unclonable function protocol responsible for quantum existencial unforgability.
This repository combines all the codes to produce the plots and results from the following article: arXiv:2404.11306

