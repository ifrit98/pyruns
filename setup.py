#!/usr/bin/python3

from distutils.core import setup

setup(name='pyruns',
      version='1.1',
      description='Python Experimentation Manager',
      author='Jason St George',
      author_email='stgeorgejas@gmail.com',
      url='https://www.github.com/ifrit98/pyruns',
      packages=['pyruns'],
      install_requires=['numpy', 'pyyaml', 'matplotlib']
     )