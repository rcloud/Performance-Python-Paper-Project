cython jac_2d.pyx
gcc -shared -fPIC -O3 -fno-strict-aliasing -I/home/rcloud/epd-7.1-1-rh5-x86_64/include/python2.7/ -I/home/rcloud/epd-7.1-1-rh5-x86_64/lib/python2.7/site-packages/numpy/core/include -o jac_2d.so jac_2d.c

