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

"""Temperature-entropy isobars for liquid water and vapour."""

# pylint: disable=invalid-name

import argparse

import numpy
from scipy.optimize import newton, brentq
from matplotlib import pyplot

from myiapws import iapws1992, iapws1995

TMIN = iapws1995.Tt
TMAX = 600 + 273.15

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='path')
path = parser.parse_args().path


## Calculations ##

# Isobar pressures and lists for the entropy and temperature arrays.
# Note: I cannot get the solution to converge at the critical pressure,
# so we use 1.01*pc instead.
pc = iapws1995.pc
ps = numpy.array([5e3, 1e5, 1e6, 5e6, 1.01*pc, 100e6, 500e6, 2000e6])  # Pa
ss = []
Ts = []

# Define a function to invert psat = p(Tsat)
# pylint: disable=redefined-outer-name
def Tsat(p):
    """Returns the saturation temperature for pressure p."""
    if p > iapws1995.pc:
        msg = 'p < pc for water vapor saturation line'
        raise ValueError(msg)
    return brentq(lambda T_: iapws1992.psat(T_)-p, iapws1995.Tt, iapws1995.Tc)

# Determine the piecewise isobars.  Obtain the gas/vapor segments first,
# followed by the liquid segments.  We don't need to obtain data in the mixed
# phase region; the plateaus emerge when the segments are concatenated.

# Gas/vapor and supercritical fluid phases
for p in ps:

    # Get the bounding temperatures for this isobar
    Tmin = Tsat(p) if p < pc else TMIN
    Tmax = TMAX

    # Get the temperature series
    T = numpy.linspace(Tmin, Tmax, 300)

    # Estimate the minimum density
    rhoest = p/(iapws1995.R*Tmax)

    # Solve for the densities along the isobar, using the previous
    # result as the next estimate
    rho = []
    for T_ in T[::-1]:
        # pylint: disable=cell-var-from-loop, undefined-loop-variable
        rho.append(newton(lambda rho_: iapws1995.p(rho_, T_)-p, rhoest))
        rhoest = rho[-1]
    rho = numpy.array(rho)[::-1]

    # Get the entropies
    s = iapws1995.s(rho, T)

    # Save the arrays
    ss.append(s)
    Ts.append(T)

# Liquid phase
for i, p in enumerate(ps):

    if p >= iapws1995.pc:
        continue

    # Get the bounding temperatures for this isobar
    Tmin = TMIN
    Tmax = Ts[i][0]

    # Get the temperature series
    T = numpy.linspace(Tmin, Tmax, 300)

    # Get the density estimate
    rhoest = iapws1992.rhosat_liquid(T[-1])

    # Solve for the densities along the isobar, using the previous
    # result as the next estimate
    rho = []
    for T_ in T[::-1]:
        rho.append(newton(lambda rho_: iapws1995.p(rho_, T_)-p, rhoest))
        rhoest = rho[-1]
    rho = numpy.array(rho)[::-1]

    # Get the entropies
    s = iapws1995.s(rho, T)

    # Concatenate the arrays
    ss[i] = numpy.concatenate((s, ss[i]))
    Ts[i] = numpy.concatenate((T, Ts[i]))

# Saturation curves: liquid-vapor
Tsat = numpy.linspace(iapws1995.Tt, iapws1995.Tc, 100)
ssat_liquid = iapws1992.ssat_liquid(Tsat)
ssat_vapor = iapws1992.ssat_vapor(Tsat)


## Plotting ##

fig = pyplot.figure(figsize=[4, 2.8])
fig.set_tight_layout(True)

# Isobars
for s, T in zip(ss, Ts):
    pyplot.plot(s/1000, T, 'k-', linewidth=1)

# Saturation curves
pyplot.plot(ssat_vapor/1000, Tsat, 'k--', linewidth=1)
pyplot.plot(ssat_liquid/1000, Tsat, 'k--', linewidth=1)

# Isobar labels
for p, s, T in zip(ps, ss, Ts):
    bbox = {'boxstyle':'square,pad=0.2', 'ec':'1', 'fc':'1'}
    rotation = 0
    if p < iapws1995.pc:
        i = numpy.searchsorted(s, iapws1995.sc)
        x = (s[i-1]+s[i])/2
        y = T[i-1]
        label = '%.0f MPa'%(p/1e6) if p/1e6 >= 1 else \
          '%.0f kPa'%(p/1e3)
    elif p == 1.01*iapws1995.pc:
        x = 6000
        y = 750
        rotation = 75
        label = r'$\mathregular{p_c}$'
    elif p == 100e6:
        x = 4326
        y = 750
        rotation = 60
        label = r'100 MPa'
    elif p == 500e6:
        x = 3644
        y = 750
        rotation = 68
        label = r'500 MPa'
    elif p == 2000e6:
        x = 3026
        y = 750
        rotation = 68
        label = r'2000 MPa'

    pyplot.text(x/1000, y, label, size=8, color='k', ha='center', va='center',
                bbox=bbox, horizontalalignment='center',
                verticalalignment='center', rotation=rotation)

pyplot.xlim(0, 10)
pyplot.ylim(TMIN, TMAX)

pyplot.xlabel(r'Entropy (kJ/K/kg)')
pyplot.ylabel(r'Temperature (K)')

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
