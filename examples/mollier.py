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

"""Mollier diagram for water."""

# pylint: disable=invalid-name

import argparse

import numpy
from scipy.optimize import newton, brentq
from matplotlib import pyplot

from myiapws import iapws1992, iapws1995

TMIN = iapws1995.Tt
TMAX = 1000 + 273.15


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='path')
path = parser.parse_args().path


# Isobars and isotherms.  Note: I cannot get the solution to converge at the
# critical pressure, so we use 1.01*pc instead.
pc = iapws1995.pc
ps = numpy.array([1e3, 1e4, 1e5, 1e6, 1.015*pc, 100e6, 500e6, 2000e6])  # Pa
Ts = numpy.array([10, 50, 100, 200, 300, 500, 800]) + 273.15  # K

# Lists for the entropy and enthalpy arrays
ss1, ss2 = [], []
hs1, hs2 = [], []

Tmins = []

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

    # Solve for the densities along the isobar, using the previous
    # result as the next estimate
    rho = []
    for T_ in T[::-1]:
        # pylint: disable=cell-var-from-loop, undefined-loop-variable
        rho.append(newton(lambda rho_: iapws1995.p(rho_, T_)-p, rhoest))
        rhoest = rho[-1]
    rho = numpy.array(rho)[::-1]

    # Get the entropies and enthalpies
    s = iapws1995.s(rho, T)
    h = iapws1995.h(rho, T)

    # Save the arrays
    ss1.append(s)
    hs1.append(h)
    Tmins.append(T[0])

# Liquid phase
for i, p in enumerate(ps):
    if p < iapws1995.pc:

        # Get the minimum and maximum temperature
        Tmin = TMIN
        Tmax = Tmins[i]

        # Get the temperature series
        T = numpy.linspace(Tmin, Tmax, 300)

        # Get the initial density estimate
        rhoest = iapws1992.rhosat_liquid(T[-1])

        # Solve for the densities along the isobar, using the previous
        # result as the next estimate
        rho = []
        for T_ in T[::-1]:
            rho.append(newton(lambda rho_: iapws1995.p(rho_, T_)-p, rhoest))
            rhoest = rho[-1]
        rho = numpy.array(rho)[::-1]

        # Get the entropies and enthalpies
        s = iapws1995.s(rho, T)
        h = iapws1995.h(rho, T)

        # Concatenate the arrays
        ss1[i] = numpy.concatenate((s, ss1[i]))
        hs1[i] = numpy.concatenate((h, hs1[i]))

# Determine the piecewise isotherms.  Obtain the gas/vapor segments first,
# followed by the liquid segments.  We don't need to obtain data in the mixed
# phase region; the plateaus emerge when the segments are concatenated.

# Gas/vapor and supercritical fluid phases
for T in Ts:

    # Get the maximum density estimate
    if T < iapws1995.Tc:
        psat = iapws1992.psat(T)
        rhoest = psat/(iapws1995.R*T)

    # Solve for the densities along the isotherm, using the previous
    # result as the next estimate.
    # pylint: disable=cell-var-from-loop, undefined-loop-variable
    rhomax = newton(lambda rho_: iapws1995.p(rho_, T)-psat, rhoest) \
      if T < iapws1995.Tc else \
      newton(lambda rho_: iapws1995.s(rho_, T)-0, iapws1995.rhoc)
    rhomin = newton(lambda x: iapws1995.s(x, T)-10000, rhoest/1e4)
    rho = numpy.logspace(numpy.log10(rhomin), numpy.log10(rhomax), 500)

    # Get the entropies and enthalpies
    s = iapws1995.s(rho, T)
    h = iapws1995.h(rho, T)

    # Save the arrays
    ss2.append(s)
    hs2.append(h)

# Liquid phase
for i, T in enumerate(Ts):

    # Get the minimum density estimate
    if T < iapws1995.Tc:
        psat = iapws1992.psat(T)
        rhoest = iapws1992.rhosat_liquid(T)
    else:
        continue

    # Solve for the densities along the isotherm, using the previous
    # result as the next estimate.
    # pylint: disable=cell-var-from-loop, undefined-loop-variable
    rhomin = newton(lambda rho_: iapws1995.p(rho_, T)-psat, rhoest)
    rhomax = newton(lambda rho_: iapws1995.s(rho_, T)-0, rhoest*1e3)
    rho = numpy.logspace(numpy.log10(rhomin), numpy.log10(rhomax), 500)

    # Get the entropies and enthalpies
    s = iapws1995.s(rho, T)
    h = iapws1995.h(rho, T)

    # Save the arrays
    ss2[i] = numpy.concatenate((ss2[i], s))
    hs2[i] = numpy.concatenate((hs2[i], h))

# Saturation curves: liquid-vapor
Tsat = numpy.linspace(iapws1995.Tt, iapws1995.Tc, 100)
ssat_liquid = iapws1992.ssat_liquid(Tsat)
ssat_vapor = iapws1992.ssat_vapor(Tsat)
hsat_liquid = iapws1992.hsat_liquid(Tsat)
hsat_vapor = iapws1992.hsat_vapor(Tsat)


## Plotting ##

fig = pyplot.figure(figsize=[4, 2.8])
fig.set_tight_layout(True)

# Isobars
for s, h in zip(ss1, hs1):
    pyplot.plot(s/1000, h/1e6, 'k-', linewidth=1)

# Isotherms
for s, h in zip(ss2, hs2):
    pyplot.plot(s/1000, h/1e6, 'k:', linewidth=1)

# Saturation curves
pyplot.plot(ssat_vapor/1000, hsat_vapor/1e6, 'k--', linewidth=1)
pyplot.plot(ssat_liquid/1000, hsat_liquid/1e6, 'k--', linewidth=1)

pyplot.xlim(0, 10)
pyplot.ylim(1, 4)

pyplot.xlabel(r'Entropy (kJ/K/kg)')
pyplot.ylabel(r'Enthalpy (MJ/kg)')


if path:
    pyplot.savefig(path)
else:
    pyplot.show()
