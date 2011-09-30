cimport numpy as np
cimport cython
from cython.parallel import *
ctypedef np.float32_t dtype_t
@cython.boundscheck(False)
@cython.wraparound(False)
def solve(np.ndarray[dtype_t, ndim=2] u, 
          np.ndarray[dtype_t, ndim=2] u_new, 
          int dimy, int dimx):
    cdef int y, x
    for y in prange(1, dimy - 1, nogil=True):
        for x in range(1, dimx - 1):
            u_new[y,x] = (u[y + 1,x] + u[y - 1,x] + u[y,x + 1] + u[y,x - 1]) / 4
            
    
