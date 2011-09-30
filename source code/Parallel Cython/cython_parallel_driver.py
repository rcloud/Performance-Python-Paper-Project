import numpy as np
import time
import sys
import jac_2d
import pickle
def initialize(u):
    u[0,:] = 100 #top row
    u[:,0] = 75 #left column
    u[:,u.shape[0] - 1] = 50 #right column


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
            jac_2d.solve(u, u_new, dim, dim)
        else:
            jac_2d.solve(u_new, u, dim, dim)
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
    start = time.time()
    output = solver(dim)
    finish = time.time()
    print finish - start
    fd = open("data.txt", "w")
    pickle.dump(output, fd)
    fd.close()

    
    