cimport numpy as np
cimport cython
from cython.parallel import *
ctypedef np.float32_t dtype_t
import numpy


def convergence_test(np.ndarray[dtype_t, ndim=2] u_new, np.ndarray[dtype_t, ndim=2] u):
    cdef double eps = .0001    
    cdef int cont = 0
    if(numpy.max(u_new - u) > eps):
        cont = 1
    return cont


@cython.boundscheck(False)
@cython.wraparound(False)
def solve(np.ndarray[dtype_t, ndim=2] u, 
           np.ndarray[dtype_t, ndim=2] u_new, 
           int dimy, int dimx):
        cdef int y, x              
        for y in prange(1, dimy - 1, nogil=True):
            for x in range(1, dimx - 1):
                u_new[y,x] = (u[y + 1,x] + u[y - 1,x] + u[y,x + 1] + u[y,x - 1]) / 4
            
    
def solver(int dim, np.ndarray[dtype_t, ndim=2] u, np.ndarray[dtype_t, ndim=2] u_new):
    cdef int iteration = 0
    cdef int cont = 1
    while cont == 1:
        if iteration % 2 == 0: #even
            solve(u, u_new, dim, dim)
        else:
            solve(u_new, u, dim, dim)
        #test for convergence            
        if iteration % 200 == 0:
            cont = convergence_test(u_new, u) 
        iteration = iteration + 1    
    return u_new