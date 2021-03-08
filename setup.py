from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("pivot.pyx", language="c++")
)

setup(
    ext_modules = cythonize("groupby.pyx", language="c++")
)