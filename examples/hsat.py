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

"""Saturation enthalpies for liquid water, ice and vapour."""

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


## Calculations ##

# Saturated liquid and vapour above triple point
T1 = numpy.linspace(iapws1995.Tt, iapws1995.Tc, 300)
hsat_liquid1 = iapws1992.hsat_liquid(T1)
hsat_vapor1 = iapws1992.hsat_vapor(T1)

# Saturated ice (below triple point)
T2 = numpy.linspace(150, iapws1995.Tt, 300)
psubl = iapws2011.psubl_ice_Ih(T2)
hsat_ice2 = iapws2006.h(T2, psubl)

# Saturated vapour below triple point
rhoest = psubl/(iapws1995.R*T2)
# pylint: disable=cell-var-from-loop
rho = numpy.array([newton(lambda rho_: iapws1995.p(rho_, T_) - p_, rhoest_)
                   for p_, T_, rhoest_ in zip(psubl, T2, rhoest)])
hsat_vapor2 = iapws1995.h(rho, T2)


## Plotting ##

fig = pyplot.figure(figsize=[4, 2.8])
fig.set_tight_layout(True)

pyplot.plot(T1, hsat_liquid1/1e6, 'k-', linewidth=1)
pyplot.plot(T1, hsat_vapor1/1e6, 'k-', linewidth=1)
pyplot.plot(T1[-1], iapws1995.hc/1e6, 'ko', markersize=4)
pyplot.plot(T2, hsat_ice2/1e6, 'k-', linewidth=1)
pyplot.plot(T2, hsat_vapor2/1e6, 'k-', linewidth=1)

pyplot.text(300, 2.2, 'Vapor', fontsize=9)
pyplot.text(400, 1.0, 'Liquid', fontsize=9)
pyplot.text(175, -0.3, 'Ice', fontsize=9)

pyplot.xlabel(r'Temperature ($\mathregular{^{\circ}C}$)')
pyplot.ylabel(r'Saturation Enthalpy (MJ/kg)')

pyplot.xlim(150, 700)

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
