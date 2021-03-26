from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("src/fastpivot/pivot/pivot.pyx", language="c++"),
    include_dirs=[numpy.get_include()]
)
