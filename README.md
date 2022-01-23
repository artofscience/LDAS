# Linear Dependency Aware Solver (LDAS)

This project contains material for using a Linear Dependency Aware Solver (LDAS) to efficiently compute the states and
sensitivities for compound structural optimisation problems.

Real-world structural optimisation problems involve multiple loading conditions and design constraints, with responses
typically depending on states of discretised governing equations. Generally, one uses gradient-based nested analysis and
design approaches to solve these problems. Herein, solving both the physical and adjoint problems dominates the overall
computational effort. Although not commonly detected, such problems can contain linear dependencies between the physical
and adjoint loads. An LDAS can detect such dependencies and avoid unnecessary solves entirely and automatically.

The project currently contains one such solver based on Gram-Schmidt orthogonalization, implemented in MATLAB and
Python. The media folder contains material from the accompanying paper.
