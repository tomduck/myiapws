#! /usr/bin/env python3

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages

DESCRIPTION = """\
IAPWS formulations for the thermodynamic properties of ordinary water.
"""

VERSION = '0.3'

setup(name='myiapws',
    version=VERSION,

    author='Thomas J. Duck',
    author_email='tomduck@tomduck.ca',
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    license='GPL',
    keywords='Thermophysical properties of water',
    url='https://github.com/tomduck/myiapws',
    download_url='https://github.com/tomduck/myiapws/tarball/'+VERSION,

    install_requires=['numpy', 'matplotlib'],

    packages=find_packages(),

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python'
        ]
    )
