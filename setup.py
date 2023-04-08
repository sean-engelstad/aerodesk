import os
from subprocess import check_output
import sys

# Numpy/mpi4py must be installed prior to installing aerodesk
import numpy
#import mpi4py

# Import distutils
from setuptools import setup, find_packages

setup(
    name="aerodesk",
    version="0.1",
    description="Learning repo for FEA and CFD",
    long_description_content_type="text/markdown",
    author="Sean P. Engelstad",
    author_email="sengeltad312@gatech.edu",
    install_requires=["numpy", "scipy>=1.2.1"],
    packages=find_packages(include=["aerodesk*"]),
)
