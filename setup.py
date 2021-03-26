from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("src/fastpivot/pivot/pivot.pyx", language="c++"),
    packages=find_packages(),
    include_dirs=[numpy.get_include(), 'src/fastpivot/pivot']
)
