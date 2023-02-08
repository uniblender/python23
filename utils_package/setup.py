from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy

setup(name="utils",
    version='1.0.0',
    url="https://github.com/uniblender/python23",
    description="Utilities for python course 2023",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    packages=find_packages(),
    )
