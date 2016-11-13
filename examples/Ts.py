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

"""Pressure-volume phase diagram for water."""

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

# Isobar pressures and lists for the entropy and temperature arrays.
# Note: I cannot get the solution to converge at the critical pressure,
# so we use 1.01*pc instead.
pc = iapws1995.pc
ps = numpy.array([5e4, 1e6, 5e6, 1.01*pc, 10*pc, 100*pc])  # Pa
ss = []
Ts = []

# Define a function to invert psat = p(Tsat)
# pylint: disable=redefined-outer-name
def Tsat(p):
    """Returns the saturation temperature for pressure p."""
    if p > iapws1995.pc:
        msg = 'p < pc for water vapor saturation line'
        raise ValueError(msg)
    return brentq(lambda x: iapws1992.psat(x)-p, iapws1995.Tt, iapws1995.Tc)

# Determine the piecewise isobars.  Obtain the gas/vapor segments first,
# followed by the liquid segments.  We don't need to obtain data in the mixed
# phase region; the plateaus emerge when the segments are concatenated.

# Gas/vapor and supercritical fluid phases
for p in ps:

    # Get the bounding temperatures for this pressure
    Tmin = Tsat(p) if p < pc else TMIN
    Tmax = TMAX

    # Get the temperature series
    T = numpy.linspace(Tmin, Tmax, 300)

    # Get the initial density estimate
    rhoest = p/(iapws1995.R*Tmax)

    # Get the densities along the isobar
    rho = []
    for T_ in T[::-1]:
        # pylint: disable=cell-var-from-loop, undefined-loop-variable
        rho.append(newton(lambda x: iapws1995.p(x, T_)-p, rhoest))
        rhoest = rho[-1]
    rho = numpy.array(rho)[::-1]

    # Get the entropies
    s = iapws1995.s(rho, T)

    # Save the arrays
    ss.append(s)
    Ts.append(T)

# Liquid phase
for i, p in enumerate(ps):
    if p < iapws1995.pc:

        # Get the minimum and maximum temperature
        Tmin = TMIN
        Tmax = Ts[i][0]

        assert Tmin >= iapws1995.Tt
        assert Tmax < iapws1995.Tc

        # Get the temperature series
        T = numpy.linspace(Tmin, Tmax, 300)

        # Get the initial density estimate
        rhoest = iapws1992.rhosat_liquid(T[-1])

        # Get the densities along the isobar
        rho = []
        for T_ in T[::-1]:
            rho.append(newton(lambda x: iapws1995.p(x, T_)-p, rhoest))
            rhoest = rho[-1]
        rho = numpy.array(rho)[::-1]

        # Get the entropies
        s = iapws1995.s(rho, T)

        # Concatenate the arrays
        ss[i] = numpy.concatenate((s, ss[i]))
        Ts[i] = numpy.concatenate((T, Ts[i]))

# Plotting

fig = pyplot.figure(figsize=[5, 3.5])
fig.set_tight_layout(True)

for s, T in zip(ss, Ts):
    pyplot.plot(s/1000, T, 'k-')

pyplot.xlim(0, 10)
pyplot.ylim(TMIN, TMAX)

pyplot.xlabel(r'Entropy (kJ/K/kg)', labelpad=6, fontsize=14)
pyplot.ylabel(r'Temperature (K)', fontsize=14)

for p, s, T in zip(ps, ss, Ts):
    if p < iapws1995.pc:
        i = numpy.searchsorted(s, iapws1995.sc)
        x = (s[i-1]+s[i])/2
        y = T[i-1]
        label = ('%.0f MPa' if p/1e6 >= 1 else '%.2f MPa') % (p/1e6)
        bbox = {'ec':'1', 'fc':'1'}
    elif p == 1.01*iapws1995.pc:
        i = numpy.searchsorted(s, iapws1995.sc)
        x = iapws1995.sc
        y = T[i]-30
        label = r'$\mathregular{p_c}$'
        bbox = {}
    elif p == 10*pc:
        x = 4000
        y = 750
        label = r'$\mathregular{10p_c}$'
        bbox = {'ec':'1', 'fc':'1'}
    elif p == 100*pc:
        x = 2750
        y = 700
        label = r'$\mathregular{100p_c}$'
        bbox = {'ec':'1', 'fc':'1'}

    pyplot.text(x/1000, y, label, size=10, color='k', ha='center', va='center',
                bbox=bbox)

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
