#!/usr/bin/env python

from distutils.core import setup

setup(
    name='Separable GrouPy',
    version='0.1',
    description='Exploiting Learned Symmetries in Group Equivaraint Convolutions. Built upon original Groupy code by Taco S. Cohen.',
    author='Attila Lengyel',
    author_email='a.lengyel@tudelft.nl',
    packages=['sepgroupy', 'sepgroupy.garray', 'sepgroupy.gconv', 'sepgroupy.gfunc'],
)
