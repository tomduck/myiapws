#! /usr/bin/env python3

# Copyright (C) 2016 Thomas J. Duck
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

"""Thermal expansion coefficient of liquid water at normal pressure between
0 and 100 C.

The IAPWS formulation allows us to calculate alpha_V(rho, T).  However, we want
to plot alpha_V(p=101.325 kPa, T).  This requires we first determine the density
(rho) that solves p(rho, T) = 101.325 kPa.  The optimization is done using
Newton's method.
"""

import numpy
from scipy.optimize import newton
from matplotlib import pyplot

from myiapws import iapws1995

# pylint: disable=invalid-name

# Define a series of temperatures
T = numpy.linspace(273.2, 373.15, 100)

# Get the saturation densities
rho0 = iapws1995.rhosat(T)[0]

# Get the density at normal pressure for this series of temperatures.
# Use the saturation densities as a first estimate.
# pylint: disable=cell-var-from-loop
rho = numpy.array([newton(lambda x: iapws1995.p(x, T_) - 101325, rho_)
                   for rho_, T_ in zip(rho0, T)])

# Get the thermal expansion coefficients
alpha_V = iapws1995.alpha_V(rho, T)


# Plotting
fig = pyplot.figure(figsize=[5, 3.5])
fig.set_tight_layout(True)
pyplot.plot(T-273.15, alpha_V, 'k-', linewidth=2)
pyplot.xlabel(r'Temperature ($\mathrm{^\circ C}$)')
pyplot.ylabel(r'Thermal Expansion Coeff ($\mathrm{K^{-1}}$)')
pyplot.show()