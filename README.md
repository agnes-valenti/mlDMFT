## GNet: Quantum impurity solvers for 1-, 2- and 3 orbitals

This github contains the trained models, as well as some examples, used to produce the results in the paper "Neural-Network Quantum Embedding Solvers for Correlated Materials".

**All trained models are in preliminary Beta-testing stage and applicability or accurate results can not be guaranteed!**

In general, we recommend using the neural-network solvers for initialization, i.e. following a GNet DMFT loop with a few CT-SEG or CT-HYB iterations.

Structure of the github repository:

mldmft/models: This folder contains all trained models with respective checkpoints. 
This includes 
1) mldmft/models/orb1: The 1-orbital Green's function solver 
2) mldmft/models/orb2: The 2-orbital Green's function solver
3) mldmft/models/orb3: The 3-orbital Green's function solver
4) mldmft/models/orb3_dens: The 3-orbital density solver

mldmft/examples: This folder contains selected examples that demonstrate how GNet is utilized in a DMFT loop and its interface with TRIQS (for the 1-and 2-orbital case).

Running the examples requires an installation of torch (>= 2.6.0) and the library TRIQS. We will update a full list of requirements, as well as examples on using the 3-orbital solver soon. 
