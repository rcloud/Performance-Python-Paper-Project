from mpi4py import MPI
import numpy as np
import time
import sys

def initialize(a):
    a[0,:] = 100 #top row
    a[:,0] = 75 #left column
    a[:,a.shape[0] - 1] = 50 #right column


def solve(a, a_new):
    dim_y = a.shape[0]
    dim_x = a.shape[1]
    for y in range(1, dim_y - 1):
        for x in range(1, dim_x - 1):
            a_new[y][x] = (a[y + 1][x] + a[y - 1][x] + a[y][x + 1] + a[y][x - 1]) / 4


if len(sys.argv) != 2:
    print "supply dimension"
    sys.exit()

dim = int(sys.argv[1])

eps = .0001

file_name = "jacobi_data.txt"
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
a = np.zeros((dim, dim), dtype=np.float32)
if rank == 0:
    initialize(a)

a_old = np.copy(a)

recv_buf = np.zeros((dim / size, dim), dtype=np.float32)
comm.Scatter(a, recv_buf)
#for exchanging data use vstack to get top and bottom rows
top = np.zeros(dim, dtype=np.float32)
bottom = np.zeros(dim, dtype=np.float32)
iteration = 0
#send top row to process above(which becomes the bottom row)
if rank != 0: #i.e. the top row cannot send to above
    comm.Send(recv_buf[0], dest=(rank - 1))
if rank != size - 1: #i.e. the bottom row cannot receive from below
    comm.Recv(bottom, source=(rank + 1))
#send bottom row to process below(this becomes top row)
if rank != size-  1: #i.e. bottom row cannot send to below
    comm.Send(recv_buf[(recv_buf.shape[0] - 1)], dest=(rank + 1))
if rank != 0: 
    comm.Recv(top, source=(rank - 1))
recv_buf_2 = np.copy(recv_buf)

#if one of the intermediary ranks
if rank != 0 and rank != size - 1:
    recv_buf_2 = np.vstack((top, recv_buf, bottom))
if rank == 0:
    recv_buf_2 = np.vstack((recv_buf, bottom))
if rank == size - 1:
    recv_buf_2 = np.vstack((top, recv_buf))

#this can be optimized out
recv_buf_new = np.copy(recv_buf_2)
num_rows = recv_buf_new.shape[0] 
"""possible problem here when some processes finish the iteration process and converge while
others don't.  the variable cont can be different for each processor """
cont = np.ones((1), dtype=np.int32)
if rank == 0:
    start = MPI.Wtime()
while cont[0] == 1:
#    print rank, iteration, cont
    if iteration % 2 == 0: #even
        if rank != 0: 
            comm.Send(recv_buf_2[1], dest=(rank - 1))
        if rank != size - 1: 
            comm.Recv(recv_buf_2[recv_buf_2.shape[0] -1], source=(rank + 1))
        if rank != size - 1: 
            comm.Send(recv_buf_2[recv_buf_2.shape[0] - 2], dest=(rank + 1))
        if rank != 0: 
            comm.Recv(recv_buf_2[0], source=(rank - 1))
        solve(recv_buf_2, recv_buf_new)
    else: #odd
        if rank != 0: 
            comm.Send(recv_buf_new[1], dest=(rank - 1))
        if rank != size - 1: 
            comm.Recv(recv_buf_new[recv_buf_2.shape[0] -1], source=(rank + 1))
        if rank != size - 1: 
            comm.Send(recv_buf_new[recv_buf_2.shape[0] - 2], dest=(rank + 1))
        if rank != 0: 
            comm.Recv(recv_buf_new[0], source=(rank - 1))    
        solve(recv_buf_new, recv_buf_2)
    #test for convergence
    #this will happen after an even, so recv_buf_new will be most recent
#    comm.Barrier()
    if iteration % 200 == 0:        
        if rank == 0: # top rank cut bottom row
            cont[0] = 0
            new_arr = recv_buf_new[0 : num_rows - 1]
            old_arr = recv_buf_2[0 : num_rows - 1]            
        elif rank == size - 1: # bottom rank cut top row
            new_arr = recv_buf_new[1 : num_rows]
            old_arr = recv_buf_2[1 : num_rows]            
        else: #middle ranks cut top and bottom
            new_arr = recv_buf_new[1 : num_rows - 1]
            old_arr = recv_buf_2[1 : num_rows - 1]            
        comm.Gather(new_arr, a)
        comm.Gather(old_arr, a_old)
        if rank == 0:
            for y in range(dim):
                for x in range(dim):
                    error = (a[y][x] - a_old[y][x]) / a[y][x]
                    if error > eps:
                        cont[0] = 1
        comm.Bcast(cont)
    iteration = iteration + 1    

if rank == 0:
    finish = MPI.Wtime()
    print iteration
    print finish - start, "seconds"

#now Gather back to rank 1
# resize the arrays so that every rank has same size. (dim / size)
if iteration % 2 == 1: 
    if rank == 0: # top rank cut bottom row
        new_arr = recv_buf_new[0 : num_rows - 1]
    elif rank == size - 1: # bottom rank cut top row
        new_arr = recv_buf_new[1 : num_rows]
    else: #middle ranks cut top and bottom
        new_arr = recv_buf_new[1 : num_rows - 1]
else:
    if rank == 0: # top rank cut bottom row
        new_arr = recv_buf_2[0 : num_rows - 1]
    elif rank == size - 1: # bottom rank cut top row
        new_arr = recv_buf_2[1 : num_rows]
    else: #middle ranks cut top and bottom
        new_arr = recv_buf_2[1 : num_rows - 1]    

#print rank
#if rank == 0:
#    print new_arr

comm.Gather(new_arr, a)

#if rank == 0:
#    print a

#    
#    comm.Scatter(a, recv_buf)

np.set_printoptions(precision=3)
#if rank == 0:
#    print(a)
#print a.shape
