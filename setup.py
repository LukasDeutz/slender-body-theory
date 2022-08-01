'''
Created on 29 Jun 2022

@author: lukas
'''
import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'slender_body_theory',
    version = '0.1',
    author = 'Lukas Deutz',
    author_email = 'scld@leeds.ac.uk',
    description = 'Implements slender body theory',
    long_description = read('README.md'),
    url = 'https://github.com/LukasDeutz/slender-body-theory.git',
    packages = find_packages()
)

                    
