import time
import sys
import jac_2d
import pickle

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "supply dimension"
        sys.exit()
    dim = int(sys.argv[1])
    start = time.time()
    output = jac_2d.solver(dim)
    finish = time.time()
    print finish - start
    fd = open("data.txt", "w")
    pickle.dump(output, fd)
    fd.close()
