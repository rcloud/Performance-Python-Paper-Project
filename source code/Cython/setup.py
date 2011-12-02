from distutils.core import setup
from distutils.extension import Extension

# numpy is required -- attempt import
try:
    import numpy
except ImportError:
    print "numpy is required"
    raise

# mpi4py is required -- attempt import
try:
    import mpi4py
except ImportError:
    print "mpi4py is required"
    raise

# cython is required -- attempt import
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print "cython is required"
    raise

include_dirs = [numpy.get_include(), '.']

ext_modules = [
    Extension(
        name="jac_2d",
        sources=["jac_2d.pyx"],
        include_dirs=include_dirs,
    ),
]

ext_modules = cythonize(ext_modules, include_path=include_dirs)
setup(
    name="jac_2d",
    packages=[],
    ext_modules=ext_modules,
)
