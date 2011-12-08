from mpi4py import MPI
from ga4py import ga
import numpy as np
import sys

EPSILON = .0001
HOW_MANY_STEPS_BEFORE_CONVERGENCE_TEST = 2
DEBUG = False

if len(sys.argv) != 2:
    print "supply dimension"
    sys.exit()

dim = int(sys.argv[1])
rank = ga.nodeid()
size = ga.nnodes()

def print_sync(obj):
    for proc in range(size):
        if rank == proc:
            print "%d: %s" % (proc, obj)
        ga.sync()

def convergence_test(g_a, g_b):
    ### PROBLEM: g_b might contain zeros which causes GA to terminate
    # subtract g_b from g_a, results stored in g_b
    ga.add(g_a, g_b, g_b, beta=-1)
    # divide g_b by g_a, results stored in g_b
    ga.elem_divide(g_b, g_a, g_b)
    # find the largets element and compare to epsilon
    value,index = ga.select_elem_max(g_b)
    if DEBUG:
        print_sync(value)
    return value < EPSILON
        
def convergence_test_L2(g_a, g_b):
    # compute L2 norm of change
    # subtract g_b from g_a, results stored in g_b
    ga.add(g_a, g_b, g_b, beta=-1)
    # compute elementwise dot product (i.e. treats N-d arrays as vectors)
    value = ga.dot(g_b, g_b)
    if DEBUG:
        print_sync(value)
    return value < EPSILON

# create GA, distribute entire rows
g_a = ga.create(ga.C_FLOAT, (dim,dim), chunk=(0,dim))
# create a duplicate GA for the convergence test
g_b = ga.duplicate(g_a)

# process 0 initializes global array
# Note: alternatively, each process could initialize its local data using
# ga.access() and ga.distribution()
a = np.zeros((dim,dim), dtype=np.float32)
if rank == 0:
    a[0,:] = 100 #top row
    a[:,0] = 75 #left column
    a[:,a.shape[0] - 1] = 50 #right column
    ga.put(g_a, a)
ga.sync()

# which piece of array do I own?
# note that rhi and chi follow python range conventions i.e. [lo,hi)
(rlo,clo),(rhi,chi) = ga.distribution(g_a)

iteration = 0
start = ga.wtime()
while True:
    iteration += 1
    if iteration % HOW_MANY_STEPS_BEFORE_CONVERGENCE_TEST == 0:
        # check for convergence will occur, so make a copy of the GA
        ga.sync()
        ga.copy(g_a, g_b)
    # the iteration
    if rlo == 0 and rhi == dim:
        # I own the top and bottom rows
        ga.sync()
        my_array = ga.access(g_a)
        my_array[1:-1,1:-1] = (
                my_array[0:-2, 1:-1] +
                my_array[2:, 1:-1] +
                my_array[1:-1,0:-2] +
                my_array[1:-1, 2:]) / 4
        ga.release(g_a)
    elif rlo == 0:
        # I own the top rows, so get top row of next domain
        next_domain_row = ga.get(g_a, (rhi,0), (rhi+1,dim))
        ga.sync()
        my_array = ga.access(g_a)
        combined = np.vstack((my_array,next_domain_row))
        my_array[1:,1:-1] = (
                combined[0:-2, 1:-1] +
                combined[2:, 1:-1] +
                combined[1:-1,0:-2] +
                combined[1:-1, 2:]) / 4
        ga.release(g_a)
    elif rhi == dim:
        # I own the bottom rows, so get bottom row of previous domain
        prev_domain_row = ga.get(g_a, (rlo-1,0), (rlo,dim))
        ga.sync()
        my_array = ga.access(g_a)
        combined = np.vstack((prev_domain_row,my_array))
        my_array[0:-1,1:-1] = (
                combined[0:-2, 1:-1] +
                combined[2:, 1:-1] +
                combined[1:-1,0:-2] +
                combined[1:-1, 2:]) / 4
        ga.release(g_a)
    else:
        # I own the middle rows, so get top and bottom row of neighboring domain
        next_domain_row = ga.get(g_a, (rhi,0), (rhi+1,dim))
        prev_domain_row = ga.get(g_a, (rlo-1,0), (rlo,dim))
        ga.sync()
        my_array = ga.access(g_a)
        combined = np.vstack((prev_domain_row,my_array,next_domain_row))
        my_array[0:,1:-1] = (
                combined[0:-2, 1:-1] +
                combined[2:, 1:-1] +
                combined[1:-1,0:-2] +
                combined[1:-1, 2:]) / 4
        ga.release(g_a)
    ga.sync()
    if iteration % HOW_MANY_STEPS_BEFORE_CONVERGENCE_TEST == 0:
        if convergence_test_L2(g_a, g_b):
            break

if DEBUG and rank == 0:
    print ga.get(g_a)

if rank == 0:
    print iteration
    print ga.wtime() - start, "seconds"
