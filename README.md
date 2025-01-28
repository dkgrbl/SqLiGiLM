# SqLiGiLM
Python module for numerical simulation of multimode squeezed light generation in lossy media



## Features

This module presents the numerical realization of results obtained in <https://arxiv.org/abs/2403.05259>:

* The solver of multimode Gaussian master equations for parametric down-conversion.
* The reconstruction of broadband modes with maximal possible squeezing from the given covariance matrix.


## Run example


To run an example:
``` make run-example ```


## Documentation

* The solver of multimode Gaussian master equations is located in the file **src/master_eq_solver.py**.

* The code for reconstructing broadband modes with maximal possible squeezing is located in the file **src/msq_basis.py**.

* The $\hbar=2$ convention is used (vacuum quadrature variance $=1$). The order of variables for the covariance matrix is $(q_1,...,q_N;p_1,...,p_N)$. 

* The function *build_MSq_basis* in the file **src/msq_basis.py** returns the basis of maximal possible squeezing. The order of sorting corresponds to the increasing minimal quadrature variance.

* The shortcoming of the function *build_MSq_basis* in **src/msq_basis.py** is the order of vectors: the occupied but unsqueezed modes are last calculated. For large matrices, it is convenient and efficient to change the order of modes and find these unsqueezed modes earlier.
This is realized in the function *build_MSq_basis_swap*, where the parameter *swap_cutoff* determines the boundary in eigenvalues for the sorting change.


## Citing

If you are using the decomposition for broadband modes with maximal possible squeezing, please cite the <https://arxiv.org/abs/2403.05259>:

Denis A. Kopylov, Torsten Meier, Polina R. Sharapova; Theory of Multimode Squeezed Light Generation in Lossy Media, arXiv:2403.05259 (2024)
