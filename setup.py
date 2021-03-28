from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        'fastpivot.pivot',
        sources = ['fastpivot/pivot.pyx'],
        language='c++',
        include_dirs=[numpy.get_include(), 'fastpivot/']
    )
]


setup(
    ext_modules = cythonize(extensions, language='c++'),
    packages = find_packages(),
    cmdclass={"build_ext": build_ext},
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'scipy>=1.6.0'
    ]
)
