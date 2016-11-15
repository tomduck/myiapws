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

"""Pressure-volume isotherms for water."""

# pylint: disable=invalid-name

import argparse

import numpy
from scipy.optimize import newton
from matplotlib import pyplot, ticker

from myiapws import iapws1992, iapws1995, iapws2006, iapws2011

PMIN = 10
PMAX = 1e8

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='path')
path = parser.parse_args().path

# Isotherm temperatures and lists for density and pressure arrays
Tc = iapws1995.Tc - 273.15  # C
Ts = numpy.array([-25, 0, 40, 100, 200, Tc, 600, 1000, 1500, 2300]) + 273.15 # K
rhos = []
ps = []

# Determine the piecewise isotherms.  Obtain the gas/vapor segments first,
# followed by the liquid and solid segments.  We don't need to obtain data
# in the mixed phase region; the plateaus emerge when the segments are
# concatenated.

# Gas/vapor and supercritical fluid phases.
for T in Ts:

    # Determine the minimum density.  Estimate it using the ideal gas law.
    # pylint: disable=cell-var-from-loop
    rhomin = newton(lambda x: iapws1995.p(x, T) - PMIN, PMIN/(iapws1995.R*T))

    # Determine the maximum density. For Tt < T < Tc estimate it using
    # the liquid-vapor saturation density from IAPWS 1992; otherwise use
    # the ideal gas law.  Convergence near the critical point requires we
    # bump up the ideal gas estiamte by a factor of 1.5.
    pmax = iapws1992.psat(T) if iapws1995.Tt < T < iapws1995.Tc else \
      iapws2011.psubl_ice_Ih(T) if T < iapws1995.Tt else PMAX
    rhoest = iapws1992.rhosat_vapor(T) if iapws1995.Tt < T < iapws1995.Tc else \
      pmax/(iapws1995.R*T)*1.5
    # pylint: disable=cell-var-from-loop
    rhomax = newton(lambda x: iapws1995.p(x, T) - pmax, rhoest)

    # Get the densities and pressures
    rho = numpy.logspace(numpy.log10(rhomin), numpy.log10(rhomax), 300)
    p = iapws1995.p(rho, T)

    # Store the data
    rhos.append(rho)
    ps.append(p)

# Liquid phase with Tt < T < Tc
for i, T in enumerate(Ts):
    if iapws1995.Tt < T < iapws1995.Tc:

        # Get the minimum and maximum pressure
        pmin = iapws1992.psat(T)  # Liquid-vapor saturation pressure
        pmax = PMAX

        # Get the pressures and densities.  Calculate successive points
        # using the prior result as the next estimate.
        p = numpy.logspace(numpy.log10(pmin), numpy.log10(pmax), 100)
        rhoest = iapws1992.rhosat_liquid(T)
        rho = []
        for p_ in p:
            # pylint: disable=cell-var-from-loop, undefined-loop-variable
            rho.append(newton(lambda x: iapws1995.p(x, T) - p_, rhoest))
            rhoest = rho[-1]
        rho = numpy.array(rho)

        # Concatenate the arrays
        ps[i] = numpy.concatenate((ps[i], p))
        rhos[i] = numpy.concatenate((rhos[i], rho))


# Solid phase
for i, T in enumerate(Ts):
    if T < iapws1995.Tt:

        # Get the minimum and maximum pressure
        pmin = iapws2011.psubl_ice_Ih(T)
        pmax = min(iapws2011.pmelt_ice_Ih(T), PMAX) if T >= 251.165 else PMAX

        # Get the pressures and densities
        p = numpy.logspace(numpy.log10(pmin), numpy.log10(pmax), 100)
        rho = iapws2006.rho(T, p)

        # Concatenate the arrays
        ps[i] = numpy.concatenate((ps[i], p))
        rhos[i] = numpy.concatenate((rhos[i], rho))


# Liquid phase with T < Tt
for i, T in enumerate(Ts):
    if T < iapws1995.Tt:  #
        # Get the minimum and maximum pressure
        pmin = iapws2011.pmelt_ice_Ih(T) if T >= 251.165 else PMAX
        pmax = PMAX

        if pmin >= PMAX:
            continue

        # Get the pressures and densities.  Calculate successive points
        # using the prior result as the next estimate.
        p = numpy.logspace(numpy.log10(pmin), numpy.log10(pmax), 100)
        rhoest = 1000  # Hard-coded estimate
        rho = []
        for p_ in p:
            rho.append(newton(lambda x: iapws1995.p(x, T) - p_, rhoest))
            rhoest = rho[-1]
        rho = numpy.array(rho)

        # Concatenate the arrays
        ps[i] = numpy.concatenate((ps[i], p))
        rhos[i] = numpy.concatenate((rhos[i], rho))

# Plotting

fig = pyplot.figure(figsize=[5, 3.5])
fig.set_tight_layout(True)

for p, T, rho in zip(ps, Ts, rhos):
    pyplot.loglog(1/rho, p, 'k-')

pyplot.xlabel(r'Volume $\mathregular{(/kg)}$', labelpad=6, fontsize=14)
pyplot.ylabel(r'Pressure', fontsize=14)

pyplot.xlim(1.e-4, 1.e5)
pyplot.ylim(10, 1e8)

ax = pyplot.gca()
ax.xaxis.set_minor_locator(ticker.FixedLocator([]))
ax.yaxis.set_minor_locator(ticker.FixedLocator([]))
ax.set_xticks([10**n for n in range(-3, 5)])
ax.set_xticklabels([r'$\mathregular{1\ L^{\ }}$', '', '',
                    r'$\mathregular{1\ m^3}$', '', '',
                    r'$\mathregular{1000\ m^3}$', ''])
ax.set_yticks([10**n for n in range(1, 8)])
ax.set_yticklabels(['', '', r'$\mathregular{1\ kPa}$', '', '',
                    r'$\mathregular{1\ MPa}$', ''])

for label in ax.xaxis.get_ticklabels():
    label.set_verticalalignment("baseline")
ax.xaxis.set_tick_params(pad=18)

for p, T, rho in zip(ps, Ts, rhos):
    if T < iapws1995.Tc:
        i = numpy.searchsorted(rho, iapws1995.rhoc)
        x = 10**(-(numpy.log10(rho[i-1]) + numpy.log10(rho[i]))/2)
        y = p[i-1]
        s = r'%.0f $\mathregular{^{\circ}C}$' % (T-273.15)
        bbox = {'ec':'1', 'fc':'1'}
    elif T == iapws1995.Tc:
        i = numpy.searchsorted(rho, iapws1995.rhoc)
        x = 1/iapws1995.rhoc
        y = p[i]/2
        s = 'Tc'
        bbox = {}
    pyplot.text(x, y, s, size=10, color='k', ha='center', va='center',
                bbox=bbox)
pyplot.text(0.8, 0.85, 'Supercritical\nisotherms:', size=10, color='k',
            ha='center', va='center', transform=ax.transAxes)
pyplot.text(0.8, 0.75, r'600 $\mathregular{^{\circ}C}$', size=10,
            color='k', ha='center', va='center', transform=ax.transAxes)
pyplot.text(0.8, 0.68, r'1000 $\mathregular{^{\circ}C}$', size=10,
            color='k', ha='center', va='center', transform=ax.transAxes)
pyplot.text(0.8, 0.61, r'1500 $\mathregular{^{\circ}C}$', size=10, color='k',
            ha='center', va='center', transform=ax.transAxes)
pyplot.text(0.8, 0.54, r'2300 $\mathregular{^{\circ}C}$', size=10,
            color='k', ha='center', va='center', transform=ax.transAxes)

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
