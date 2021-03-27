from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        'pivot',
        sources = ['src/fastpivot/pivot.pyx'],
        include_dirs=[numpy.get_include(), 'src/fastpivot/']
    )
]


setup(
    ext_modules = cythonize(extensions),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'scipy>=1.6.0'
    ]
)
