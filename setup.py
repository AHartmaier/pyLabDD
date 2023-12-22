#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup

test_requirements = ['pytest>=3', ]
setup_requirements = ['pytest-runner', ]
with open('README.md') as readme_file:
    readme = readme_file.read()

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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="Python Laboratory for Dislocation Dynamics",
    install_requires=['numpy', 'matplotlib', 'pytest', 'fmodpy'],
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='Dislocation Dynamics',
    name='pylabdd',
    packages=['pylabdd'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AHartmaier/pyLabDD',
    version='1.1',
    zip_safe=False,
)
