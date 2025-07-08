**Research Task: *Simulations and Tests of Higher-Spin Ultra-Light Dark Matter*.** \
This project focuses on creating and refining simulation tools for axion-like particle dark matter. The aim is to simulate and test models of ultra-light dark matter, particularly those involving higher-spin particles. The work involves:

Reviewing literature on the dark matter problem, emphasizing ultra-light dark matter models.
Utilizing and modifying computational tools for simulating cosmological structures of ultra-light dark matter.
Developing new simulation approaches tailored to ultra-light dark matter.
Extracting valuable insights from simulations and defining observational tests.
The simulations aim to advance our understanding of the properties and cosmological implications of ultra-light dark matter, contributing to broader dark matter research efforts.


**Phase 0:**
- making github repo with functional cupy - ✓ done
  - it should include requirements.txt needed to run the main.py file
- Solving Schrödinger equation - in vacuum - using furier transform - ✓ done 24.1.
- Extending the solution to 2D


**Phase 0.5:**
- making 3D waves possible ✓
- adding error handling into wave function class X - questionable
- Unifying functions for 1D,2D,3D animation X

**Phase 1:**
- solving Schrödinger equation with harmonic potential ✓
- programming analytical solution for LHO Schr. system and comparing the two  ✓
- monitoring the errors ✓ 
  - the error of my solution from the analytical one ✓ - it should be aight
  - the error of time propagation of the probability density  ✓ - also should be fine
  
**Phase 2**
- Poisson-Schrödinger System implemented using the split-step method
- Implementation is based on https://arxiv.org/pdf/2101.01828.
- Adding a chance to include gravitational potential in Wave function ✓
- Calculating the density of the wave function in given time of space -> transforming it to k_space and using it to solve the poisson equation using cuda fft ✓
- using the resulting potential to evolve the wave function in split step methode ✓

**Phase 2.5**
- Changing the structure of the code to allow for multiple wave functions systems ✓
- Evolution is now handeled from within Simulation_class ✓
- added possibility to include multiple wave_function_class instances in one Simulation ✓
- testing functionality of packet creation - as of 3.4.2025 it should be possible to create LHO packet (based on analytical solution), gaussian packet, costum packet read from a file ✓
- Adding momentum to wave functions actually makes them move ✓

**Phase 2.5.1**
- testing various system configurations and how they evolve with gravity ✓
- using sperically symmetric wave function packet a test was run to see if it behaves as expected - it does ✓

**Phase 3**
- adding physical units to the simulation - ✓

**Phase 3.5**
- ran a simulation to verify correct implementation of physical units by simulating circular orbit around a massive object ✓

**Phase 4**
- Creation of Wave_vector_class which allows for implementation of different spins of wave functions using Clebsh Gordan coeffs ✓
- streamlined the creation of wave function so that it is no longer needed to create a Wave_function instance to create a Wave_vector
instance but one should create Wave_vector instance with params of Wave_fucntion and spin ✓
- allowed for mass scaling of the soliton using methodes from https://arxiv.org/pdf/2502.03561 ✓

**Phase 5**
- Added tracking of kinetic and potential energy along the simulation ✓
- added plotting.py which allows to plot from snapshots ✓
- Implementation of dark matter profile calculation - working on it
