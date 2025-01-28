#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####




import numpy as np

##########
##########  
##########

def make_eigendecomp_h(matrix, order="decreasing"):
    eigen_vals, eigen_vecs = np.linalg.eigh(matrix)
    if order=="decreasing":
        idx = np.real(eigen_vals).argsort()[::-1] 
    elif order=="increasing":
        idx = np.real(eigen_vals).argsort() 
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:,idx]
    return eigen_vals, eigen_vecs
