import numpy as np
import time
import sys
import jac_2d
import pickle
def initialize(u):
    u[0,:] = 100 #top row
    u[:,0] = 75 #left column
    u[:,u.shape[0] - 1] = 50 #right column





def setup(dim):
    u = np.zeros((dim, dim), dtype=np.float32)
    initialize(u)
    u_new = np.copy(u)
    u_new = jac_2d.solver(dim, u, u_new)
    return u_new

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "supply dimension"
        sys.exit()
    dim = int(sys.argv[1])
    start = time.time()
    output = setup(dim)
    finish = time.time()
    print finish - start
    fd = open("data.txt", "w")
    pickle.dump(output, fd)
    fd.close()

    
    