# DMRG_demo
Small demonstration of DMRG applied to the Ising model.

Minimal two-site DMRG for the 1D Transverse-Field Ising Model (OBC)
with comparison to exact diagonalization (ED) for small N (energies and expectation values of Z and X operators).

Pedagogical implementation - not optimized.
It shows the essential steps in DMRG: building an MPO, sweeping with two-site
updates (solve effective eigenproblem, SVD + truncate), and evaluating
observables and energy. Comparison is then made to ED results.
