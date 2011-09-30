import numpy as np
import time
import sys
import random
from mpi4py import MPI
import pickle
import os
import time
def solver(u, u_new):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    eps = .0001
    cont = True
    cont_vals = comm.allgather(cont)
    iteration = 0
    while cont_vals[-1] == True: # after sorting, True will be last element unless all are false
        if iteration % 2 == 0: #even
            if rank != 0:
                comm.Send(u[1], dest=(rank - 1)) # send top row to proc above
            if rank != size - 1:
                comm.Recv(u[-1], source=(rank + 1)) # recv boundary from proc below
            if rank != size - 1:
                comm.Send(u[-2], dest=(rank + 1)) # send bottom row to proc below
            if rank != 0:
                comm.Recv(u[0], source=(rank - 1)) # recv top row from proc above
            u_new[1:-1, 1:-1] = ((u[0:-2, 1:-1] + u[2:, 1:-1]) + (u[1:-1,0:-2] + u[1:-1, 2:])) / 4
        else:
            if rank != 0:
                comm.Send(u_new[1], dest=(rank - 1)) # send top row to proc above
            if rank != size - 1:
                comm.Recv(u_new[-1], source=(rank + 1)) # recv boundary from proc below
            if rank != size - 1:
                comm.Send(u_new[-2], dest=(rank + 1)) # send bottom row to proc below
            if rank != 0:
                comm.Recv(u_new[0], source=(rank - 1)) # recv top row from proc above
            u[1:-1, 1:-1] = ((u_new[0:-2, 1:-1] + u_new[2:, 1:-1]) + (u_new[1:-1,0:-2] + u_new[1:-1, 2:])) / 4
        #test for convergence
        if iteration % 1000 == 0:
            cont = False
            if(np.max(u_new - u) > eps):
                cont = True
            cont_vals = comm.allgather(cont)
            cont_vals.sort()
        iteration = iteration + 1    
    return iteration

def initialize(dim, comm, rank, size):
    init_vals = None    
    if rank == 0:
        #top = random.random() * 100
        #left = random.random() * 75
        #right = random.random() * 50
        #bottom = random.random() * 25
        top = 100
        left = 75
        right = 50
        bottom = 25
        init_vals = {'top':top, 'left':left, 'right':right, 'bottom':bottom}    
    init_vals = comm.bcast(obj=init_vals)
    u = np.zeros((dim / size, dim))    
    if rank == 0: 
        u = np.vstack((u, np.zeros(dim)))
        u[0, :] = init_vals['top']
        u[:, 0] = init_vals['left']
        u[:, -1] = init_vals['right']

    elif rank == size - 1:
        u = np.vstack((np.zeros(dim), u))
        u[-1, :] = init_vals['bottom']
        u[:, 0] = init_vals['left']
        u[:, -1] = init_vals['right']
    else:
        u = np.vstack((np.zeros(dim), u, np.zeros(dim)))
        u[:, 0] = init_vals['left']
        u[:, -1] = init_vals['right']
        
    u_new = np.copy(u)
    return u, u_new

def setup(dim):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(1):
        if rank == 0:
            log_file = open("/home/rcloud/Dropbox/projects/unsteady_laplace/output/log.txt", "a")
            log_file.write("\n****")
            log_file.write("\ndimensions: " + str(dim))
            log_file.write("\nNum Procs: " + str(size))
            log_file.write("\nrun number " + str(i))
            start = time.time()
        u, u_new = initialize(dim, comm, rank, size)
        iterations = solver(u, u_new)
        if rank == 0:
            finish = time.time()
            log_file.write("\nRuntime: " + str(finish - start) + "seconds")
            log_file.write("\nIterations = " + str(iterations))
            log_file.write("\n")
        if rank < 10:
            f_name = "/home/rcloud/Dropbox/projects/unsteady_laplace/output/data_0%s.txt" % str(rank)
        else:   
            f_name = "/home/rcloud/Dropbox/projects/unsteady_laplace/output/data_%s.txt" % str(rank)
        f = open(f_name, "w")
        pickle.dump(u, f)
        f.close()
            
    if rank == 0:
        log_file.close()
    return
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "supply dimension"
        sys.exit()
    dim = int(sys.argv[1])
    setup(dim)
