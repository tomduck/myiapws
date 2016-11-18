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

"""Saturation enthalpies for liquid water, ice and vapour.

Similar to cp.py, determines the vapor densities (rho) that solves
p(rho, T) = psat for temperatures below the triple point.  The optimization
is done using Newton's method.  This allows the vapor enthalpy over ice
to be calculated.

The other enthalpies require nothing fancy.  Functions for determining
saturation enthalpies above the triple point already exist in iapws1992.
For ice we simply must evaluate the sublimation pressure (from iapws2011) and
calculate the enthalpy from it (using iapws2006).
"""

# pylint: disable=invalid-name

import argparse

import numpy
from matplotlib import pyplot
from scipy.optimize import newton

from myiapws import iapws1992, iapws1995, iapws2006, iapws2011


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='path')
path = parser.parse_args().path


# Define a series of temperatures
T1 = numpy.linspace(iapws1995.Tt, iapws1995.Tc, 300)  # Liquid
T2 = numpy.linspace(150, iapws1995.Tt, 300)  # Liquid


# Calculations

# Vapor over liquid
hsat_liquid1 = iapws1992.hsat_liquid(T1)
hsat_vapor1 = iapws1992.hsat_vapor(T1)

# Vapor over ice
psubl = iapws2011.psubl_ice_Ih(T2)
hsat_ice2 = iapws2006.h(T2, psubl)

rho0  = psubl/(iapws1995.R*T2)
rho = numpy.array([newton(lambda x: iapws1995.p(x, T_) - p_, rho_) for p_, T_, rho_ in zip(psubl, T2, rho0)])
hsat_vapor2 = iapws1995.h(rho, T2)


# Plotting

fig = pyplot.figure(figsize=[5, 3.5])
fig.set_tight_layout(True)

pyplot.plot(T1, hsat_liquid1/1e6, 'k-', linewidth=2)
pyplot.plot(T1, hsat_vapor1/1e6, 'k-', linewidth=2)
pyplot.plot(T1[-1], iapws1995.hc/1e6, 'ko')
pyplot.plot(T2, hsat_ice2/1e6, 'k-', linewidth=2)
pyplot.plot(T2, hsat_vapor2/1e6, 'k-', linewidth=2)

pyplot.text(300, 2.2, 'Vapor')
pyplot.text(400, 1.0, 'Liquid')
pyplot.text(175, -0.3, 'Ice')

pyplot.xlabel(r'Temperature ($\mathregular{^{\circ}C}$)', fontsize=14)
pyplot.ylabel(r'(MJ/kg)', fontsize=14)
title = pyplot.title('Saturation Enthalpy')
title.set_position([0.5, 1.03])

pyplot.xlim(150, 700)

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
