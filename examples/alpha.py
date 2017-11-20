#! /usr/bin/env python3

# Copyright (C) 2016-2017 Thomas J. Duck
#
# Thomas J. Duck <tomduck@tomduck.ca>
# Department of Physics and Atmospheric Science, Dalhousie University
# PO Box 15000 | Halifax NS B3H 4R2 Canada
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Isobaric thermal expansion coefficient versus temperature for liquid water
at normal pressure."""

# pylint: disable=invalid-name

import argparse

import numpy
from scipy.optimize import newton
from matplotlib import pyplot

from myiapws import iapws1992, iapws1995


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='path')
path = parser.parse_args().path


## Calculations ##

# Define a series of temperatures
T = numpy.linspace(273.2, 373.15, 100)

# Determine the density at normal pressure for this series of temperatures.
# Use the liquid saturation density for each temperature as a first estimate.
# pylint: disable=cell-var-from-loop
rhoest = iapws1992.rhosat_liquid(T)
rho = numpy.array([newton(lambda rho_: iapws1995.p(rho_, T_) - 101325, rhoest_)
                   for rhoest_, T_ in zip(rhoest, T)])

# Get the thermal expansion coefficients
alpha = iapws1995.alpha(rho, T)


## Plotting ##

fig = pyplot.figure(figsize=[4, 2.8])
fig.set_tight_layout(True)

pyplot.plot(T-273.15, alpha*1000, 'k-', linewidth=1)
pyplot.plot([-5, 105], [0, 0], 'k:', linewidth=1)

pyplot.xlim(-5, 105)

pyplot.text(33, 2.6, '101.325 kPa', size=9)

pyplot.xlabel(r'Temperature ($\mathrm{^{\circ}C}$)')
pyplot.ylabel('Thermal Expansion\nCoefficient ($\\%/10\\ \\mathrm{K}$)')

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
