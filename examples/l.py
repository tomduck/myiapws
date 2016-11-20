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

"""Enthalpies of transformation.

This repeats the approach of h.py in calculating the enthalpies.
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
Tt = iapws1995.Tt
Tc = iapws1995.Tc
T1 = numpy.linspace(Tt, Tc, 300)       # Vapor over liquid
T2 = numpy.linspace(150, Tt, 300)      # Vapor over ice
T3 = numpy.linspace(251.165, Tt, 300)  # Liquid over ice


# Calculations

# Vapor over liquid
hsat_liquid1 = iapws1992.hsat_liquid(T1)
hsat_vapor1 = iapws1992.hsat_vapor(T1)
Lvap = hsat_vapor1 - hsat_liquid1

# Vapor over ice
psubl = iapws2011.psubl_ice_Ih(T2)
hsat_ice2 = iapws2006.h(T2, psubl)

# pylint: disable=cell-var-from-loop
rho0 = psubl/(iapws1995.R*T2)
rho = numpy.array([newton(lambda x: iapws1995.p(x, T_) - p_, rho_) \
                   for p_, T_, rho_ in zip(psubl, T2, rho0)])
hsat_vapor2 = iapws1995.h(rho, T2)

Lsub = hsat_vapor2 - hsat_ice2

# Liquid water over ice
pmelt = iapws2011.pmelt_ice_Ih(T3)
hsat_ice3 = iapws2006.h(T3, pmelt)

# pylint: disable=cell-var-from-loop
rho = numpy.array([newton(lambda x: iapws1995.p(x, T_) - p_, 1000) \
                   for p_, T_ in zip(pmelt, T3)])
hsat_liquid3 = iapws1995.h(rho, T3)

Lfus = hsat_liquid3 - hsat_ice3


# Plotting

fig = pyplot.figure(figsize=[5, 3.5])
fig.set_tight_layout(True)

pyplot.plot(T1, Lvap/1e6, 'k-', linewidth=2)
pyplot.plot(T2, Lsub/1e6, 'k-', linewidth=2)
pyplot.plot(T3, Lfus/1e6, 'k-', linewidth=2)

pyplot.xlabel(r'Temperature ($\mathregular{^{\circ}C}$)', fontsize=14)
pyplot.ylabel(r'(MJ/kg)', fontsize=14)
title = pyplot.title('Enthalpy of Transformation')
title.set_position([0.5, 1.03])

pyplot.xlim(150, 700)
pyplot.ylim(0, 3)

pyplot.text(470, 1.5, r'$L_\mathrm{vap}$', fontsize=14)
pyplot.text(185, 2.5, r'$L_\mathrm{sub}$', fontsize=14)
pyplot.text(220, 0.5, r'$L_\mathrm{fus}$', fontsize=14)

pyplot.plot([Tt, Tt], [0, 3], 'k:')
pyplot.text(Tt, 1.5, r'$T_\mathrm{tp}$', color='k', fontsize=14,
            ha='center', va='center', bbox={'ec':'1', 'fc':'1'})

ax = pyplot.gca()
pyplot.plot([Tc, Tc], [0, 3], 'k:')
pyplot.text(Tc, 1.5, r'$T_\mathrm{c}$', color='k', fontsize=14,
            ha='center', va='center', bbox={'ec':'1', 'fc':'1'})

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
