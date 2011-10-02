# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

cimport cython
cimport numpy as np
from cython.parallel cimport prange

import numpy as np

ctypedef np.float32_t dtype_t

def initialize(u):
    u[0,:] = 100 #top row
    u[:,0] = 75 #left column
    u[:,u.shape[0] - 1] = 50 #right column

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint convergence_test(dtype_t[:, ::1] u_new, dtype_t[:, ::1] u) nogil:
    eps = .0001
    cdef Py_ssize_t i, j
    for i in xrange(u.shape[0]):
        for j in xrange(u.shape[1]):
            if u_new[i, j] - u[i, j] > eps:
                return True

    return False

def solver(int dim):
    array = np.zeros((dim, dim), dtype=np.float32)
    initialize(array)

    cdef dtype_t[:, ::1] u = array
    cdef dtype_t[:, ::1] u_new = array.copy()

    cdef bint cont = True
    cdef int iteration = 0
    with nogil:
        while cont:
            if iteration % 2 == 0: #even
                solve(u, u_new, dim, dim)
            else:
                solve(u_new, u, dim, dim)

            #test for convergence
            if iteration % 200 == 0:
                cont = convergence_test(u_new, u)

            iteration = iteration + 1

    # return the original u_new numpy array object, not a cython.memoryview
    return u_new.base

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve(dtype_t[:, ::1] u, dtype_t[:, ::1] u_new,
                int dimy, int dimx) nogil:
    cdef int y, x
    for y in prange(1, dimy - 1):
        for x in range(1, dimx - 1):
            u_new[y,x] = (u[y + 1,x] + u[y - 1,x] + u[y,x + 1] + u[y,x - 1]) / 4
