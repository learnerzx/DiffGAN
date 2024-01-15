from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(
    ext_modules = cythonize("Masks/direct/poisson.pyx"),
    include_dirs = [np.get_include()]
)
#D:\Anaconda3\envs\pytorch1\Lib\site-packages\numpy\core\include\numpy