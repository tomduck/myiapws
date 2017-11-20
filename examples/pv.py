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

"""Pressure-volume isotherms for liquid water, vapour and ice."""

# pylint: disable=invalid-name

import argparse

import numpy
from scipy.optimize import newton
from matplotlib import pyplot, ticker

from myiapws import iapws1992, iapws1995, iapws2006, iapws2011

PMIN = 10
PMAX = 1e9


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='path')
path = parser.parse_args().path


## Calculations ##

# Isotherm temperatures and lists for density and pressure arrays
Tc = iapws1995.Tc - 273.15  # C
Ts = numpy.array([-25, 0, 40, 100, 200, Tc, 1000, 2300]) + 273.15 # K
rhos = []
ps = []

# Determine the piecewise isotherms.  Obtain the gas/vapor segments first,
# followed by the liquid and solid segments.  We don't need to obtain data
# in the mixed phase region; the plateaus emerge when the segments are
# concatenated.

# Gas/vapor and supercritical fluid phases
for T in Ts:

    # Determine the minimum density.  Estimate it using the ideal gas law.
    # pylint: disable=cell-var-from-loop
    rhomin = newton(lambda x: iapws1995.p(x, T) - PMIN, PMIN/(iapws1995.R*T))

    # Determine the maximum density. For Tt < T < Tc use the liquid-vapor
    # saturation density from IAPWS 1992; otherwise solve for it using an
    # estimate from the ideal gas law (the extra factor of 1.5 helps avoid
    # convergence problems around the critical density).
    if iapws1995.Tt < T < iapws1995.Tc:
        rhomax = iapws1992.rhosat_vapor(T)
    else:
        pmax = iapws2011.psubl_ice_Ih(T) if T < iapws1995.Tt else PMAX
        rhoest = pmax/(iapws1995.R*T)*1.5
        # pylint: disable=cell-var-from-loop
        rhomax = newton(lambda rho_: iapws1995.p(rho_, T) - pmax, rhoest)

    # Get the densities and pressures
    rho = numpy.logspace(numpy.log10(rhomin), numpy.log10(rhomax), 300)
    p = iapws1995.p(rho, T)

    # Store the data
    rhos.append(rho)
    ps.append(p)

# Liquid and solid phases
for i, T in enumerate(Ts):

    if iapws1995.Tt < T < iapws1995.Tc:  # Liquid

        # Get the minimum and maximum pressure
        pmin = iapws1992.psat(T)  # Liquid-vapor saturation pressure
        pmax = PMAX

        # Get the pressures and densities.  Calculate successive points
        # using the previous result as the next estimate.
        p = numpy.logspace(numpy.log10(pmin), numpy.log10(pmax), 100)
        rhoest = iapws1992.rhosat_liquid(T)
        rho = []
        for p_ in p:
            # pylint: disable=cell-var-from-loop, undefined-loop-variable
            rho.append(newton(lambda rho_: iapws1995.p(rho_, T) - p_, rhoest))
            rhoest = rho[-1]
        rho = numpy.array(rho)

        # Concatenate the arrays
        ps[i] = numpy.concatenate((ps[i], p))
        rhos[i] = numpy.concatenate((rhos[i], rho))

    elif T < iapws1995.Tt:  # Solid

        # Get the minimum and maximum pressure
        pmin = iapws2011.psubl_ice_Ih(T)
        pmax = min(iapws2011.pmelt_ice_Ih(T), PMAX) if T >= 251.165 else PMAX

        # Get the pressures and densities
        p = numpy.logspace(numpy.log10(pmin), numpy.log10(pmax), 100)
        rho = iapws2006.rho(T, p)

        # Concatenate the arrays
        ps[i] = numpy.concatenate((ps[i], p))
        rhos[i] = numpy.concatenate((rhos[i], rho))

# Saturation curves: liquid-vapor
Ts_ = numpy.linspace(iapws1995.Tt, iapws1995.Tc, 100)
psat = iapws1992.psat(Ts_)
rhosat_vapor = iapws1992.rhosat_vapor(Ts_)
rhosat_liquid = iapws1992.rhosat_liquid(Ts_)

# Saturation curves: solid-vapor
Ts_ = numpy.linspace(iapws1995.Tt-50, iapws1995.Tt, 100)
psat2 = iapws2011.psubl_ice_Ih(Ts_)
# pylint: disable=cell-var-from-loop
rhosat_vapor2 = numpy.array([newton(lambda x: iapws1995.p(x, T) - p, \
                             p/(iapws1995.R*T)) for T, p in zip(Ts_, psat2)])
rhosat_solid2 = iapws2006.rho(Ts_, p)


## Plotting ##

fig = pyplot.figure(figsize=[4, 2.8])
fig.set_tight_layout(True)

# Isotherms
for p, T, rho in zip(ps, Ts, rhos):
    pyplot.loglog(1/rho, p, 'k-', linewidth=1)

# Saturation curves
pyplot.loglog(1/rhosat_vapor, psat, 'k--', linewidth=1)
pyplot.loglog(1/rhosat_liquid, psat, 'k--', linewidth=1)
pyplot.loglog(1/rhosat_vapor2, psat2, 'k--', linewidth=1)
pyplot.loglog(1/rhosat_solid2, psat2, 'k--', linewidth=1)

# Axis labels
pyplot.xlabel(r'Volume $\mathregular{(/kg)}$')
pyplot.ylabel(r'Pressure')

# Axis limits
pyplot.xlim(1.e-4, 1.e5)
pyplot.ylim(PMIN, PMAX)

# Ticks and labels
ax = pyplot.gca()
ax.xaxis.set_minor_locator(ticker.FixedLocator([]))
ax.yaxis.set_minor_locator(ticker.FixedLocator([]))
ax.set_xticks([10**n for n in range(-3, 5)])
ax.set_xticklabels([r'$\mathregular{1\ L^{\ }}$', '', '',
                    r'$\mathregular{1\ m^3}$', '', '',
                    r'$\mathregular{1000\ m^3}$', ''])
ax.set_yticks([10**n for n in range(2, 9)])
ax.set_yticklabels(['', r'$\mathregular{1\ kPa}$', '', '',
                    r'$\mathregular{1\ MPa}$', '', ''])
for label in ax.xaxis.get_ticklabels():
    label.set_verticalalignment("baseline")
ax.xaxis.set_tick_params(pad=12)

# Isotherm labels
for p, T, rho in zip(ps, Ts, rhos):
    if T < iapws1995.Tc:
        i = numpy.searchsorted(rho, iapws1995.rhoc)
        x = 10**(-(numpy.log10(rho[i-1]) + numpy.log10(rho[i]))/2)
        y = p[i-1]
        s = r'%.0f $\mathregular{^{\circ}C}$' % (T-273.15)
        bbox = {'boxstyle':'square,pad=0.2', 'ec':'1', 'fc':'1'}
        pyplot.text(x, y*0.9, s, size=8, color='k', ha='center', va='center',
                    bbox=bbox)
ax.annotate(r'$T_\mathrm{c}$', xy=(0.012, 2e7), xytext=(0.03, 2e8),
            arrowprops=dict(facecolor='black', shrink=0.0, width=0.1,
                            headwidth=2, headlength=2), size=8)
ax.annotate(r'1000 $\mathregular{^{\circ}C}$', xy=(0.08, 0.9e7),
            xytext=(0.16, 5e7),
            arrowprops=dict(facecolor='black', shrink=0.0, width=0.1,
                            headwidth=2, headlength=2), size=8)
ax.annotate(r'2300 $\mathregular{^{\circ}C}$', xy=(0.66, 2.4e6),
            xytext=(1, 7.2e6),
            arrowprops=dict(facecolor='black', shrink=0.0, width=0.1,
                            headwidth=2, headlength=2), size=8)


if path:
    pyplot.savefig(path)
else:
    pyplot.show()
