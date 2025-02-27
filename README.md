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
- adding error handling into wave function class X
- Unifying functions for 1D,2D,3D animation X

**Phase 1:**
- solving Schrödinger equation with harmonic potential ✓
- monitoring the errors ✓ 
  - the error of my solution from the analytical one ✓
  - the error of time propagation of the probability density  ✓
  
**Phase 2**

- Poisson-Schrödinger System implemented using the split-step method
- Implementation is based on https://arxiv.org/pdf/2101.01828.
- Adding a chance to include gravitational potential in Wave function ✓
- Calculating the density of the wave function in given time of space -> transforming it to k_space and using it to solve the poisson equation using cuda fft ✓
- using the resulting potential to evolve the wave function in split step methode ✓
