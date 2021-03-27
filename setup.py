from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="fastpivot.pivot", 
        sources=["src/fastpivot/pivots/pivot.pyx"],
        include_dirs=[numpy.get_include(), "src/fastpivot/pivots/"]
    )
]

setup(
    ext_modules = cythonize(extensions),#cythonize("src/fastpivot/pivots/pivot.pyx", language="c++"),
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0'
        'scipy>=1.6.0'
    ]
)
