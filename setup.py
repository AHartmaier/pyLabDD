#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from numpy.distutils.core import setup, Extension
#from setuptools import find_packages

lib = Extension(name='pylabdd.pkforce', sources=['pylabdd/PK_force.F90'])

with open('README.md') as readme_file:
    readme = readme_file.read()

test_requirements = ['pytest>=3', ]
setup_requirements = ['pytest-runner', ]

setup(
    author="Alexander Hartmaier",
    author_email='alexander.hartmaier@rub.de',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Python Laboratory for Dislocation Dynamics",
    install_requires=['numpy', 'matplotlib'],
    ext_modules = [lib],
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='Dislocation Dynamics',
    name='pylabdd',
    packages=['pylabdd'], #find_packages(exclude=["*tests*"]),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AHartmaier/pyLabDD',
    version='1.0',
    zip_safe=False,
)
