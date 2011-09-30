"""
Author: Robert L Cloud, rcloud@gmail.com
http://www.robertlouiscloud.com

Created: 19/08/2011

This is a simple solver of the Laplace equation
in two dimensions using Jacobian iteration. 
"""

import pyopencl as cl
import numpy as np
import pickle
import time
import sys

def get_kernel(size):
    width_macro = "#define WIDTH %s\n" %str(size)
    index_macro = "#define TD(u, y, x) (u[(y) * WIDTH + (x)])\n"
    kernel_source = width_macro + index_macro + """
    __kernel void solve(__global float *u, 
                    __global float *u_new)
                    {

                    int id = get_global_id(0);

                    int y = id / WIDTH;
                    int x = id % WIDTH;

                    if (y != 0 && y != WIDTH - 1 && x != 0 && x != WIDTH - 1)
                    {
                        TD(u_new, y, x)   = (TD(u, y + 1, x) +
                                            TD(u, y - 1, x) + 
                                            TD(u, y, x + 1) +
                                            TD(u, y, x - 1)) / 4;
                    }
                    }
    """
    return kernel_source

def initialize(a):
    #initialized using Chapra's values
    a[0,:] = 100 #top row
    a[:,0] = 75 #left column
    a[:,a.shape[0] - 1] = 50 #right column

def convergence_test(u_new, u):
    eps = .0001    
    cont = False
    if(np.max(u_new - u) > eps):
        cont = True
    return cont
    
def solver(size):
    """
    size is the m=n domain to be solved
    """
    ctx = cl.create_some_context()
    """
    for Jacobian iteration we need two arrays, one to store the 
    values from timestep i and one for timestep values i+1
    """
    u = np.zeros((size,size), dtype=np.float32)
    initialize(u)
    u_new=np.copy(u)
    program = cl.Program(ctx, get_kernel(size)).build()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    #create the memory objects on the device
    u_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
    u_new_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u_new)
    start = time.time()
    cont = True
    iteration = 0
    while cont == True:
        if iteration % 2 == 0:
            program.solve(queue, (size * size,), None, u_dev, u_new_dev)
        else:
            program.solve(queue, (size * size,), None, u_new_dev, u_dev)
        if iteration % 200 == 0:
            cl.enqueue_copy(queue, u_new, u_new_dev)
            cl.enqueue_copy(queue, u, u_dev)
            cont = convergence_test(u_new, u)
        iteration = iteration + 1
    cl.enqueue_copy(queue, u_new, u_new_dev)
    finish = time.time()
    print finish - start
    return u_new

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "supply dimension"
        sys.exit()
    dim = int(sys.argv[1])
    output = solver(dim)
    fd = open("data.txt", "w")
    pickle.dump(output, fd)
    fd.close()
