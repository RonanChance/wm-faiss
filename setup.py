#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='wm_faiss',
    version='1.0.0',
    description= "Classifiers/Estimators using FAISS for efficiency",
    long_description='See: https://github.com/RonanChance/faiss-wm',
    author='Ronan Donovan',
    author_email='rcdonovan@wm.edu',
    url='https://github.com/RonanChance/wm-faiss',
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'numpy>=1.16.0',
        'scikit-learn>=0.22.0',
        'faiss-cpu>=1.7.0'
    ]
)