import numpy as np
import time
import sys

def initialize(u):
    u[0,:] = 100 #top row
    u[:,0] = 75 #left column
    u[:,u.shape[0] - 1] = 50 #right column


def solve(u, u_new):
    dim_y = u.shape[0]
    dim_x = u.shape[1]
    for y in range(1, dim_y - 1):
        for x in range(1, dim_x - 1):
            u_new[y,x] = (u[y + 1,x] + 
                           u[y - 1,x] + 
                           u[y,x + 1] + 
                           u[y,x - 1]) / 4

def convergence_test(u_new, u):
    eps = .0001    
    cont = False
    if(np.max(u_new - u) > eps):
        cont = True
    return cont

def solver(dim):
    u = np.zeros((dim, dim), dtype=np.float32)
    initialize(u)
    u_new = np.copy(u)

    cont = True
    iteration = 0
    while cont == True:
        if iteration % 2 == 0: #even
            solve(u, u_new)
        else:
            solve(u_new, u)
        #test for convergence
        if iteration % 200 == 0:
            cont = convergence_test(u_new, u) 
        iteration = iteration + 1    
    return u_new

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "supply dimension"
        sys.exit()
    dim = int(sys.argv[1])
    f = open("straight_python.txt", "a")
    start = time.time()
    output = solver(dim)
    finish = time.time()
    f.write(str(dim) + "\t" + str(finish - start) + "\n")
    f.close()
    
