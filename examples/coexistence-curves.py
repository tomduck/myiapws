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

"""Pressure-temperature coexistence curves for water."""

# pylint: disable=invalid-name

import argparse

import numpy
from matplotlib import pyplot, ticker

from myiapws.iapws2011 import pmelt_ice_Ih, pmelt_ice_III, pmelt_ice_V
from myiapws.iapws2011 import pmelt_ice_VI, pmelt_ice_VII, psubl_ice_Ih
from myiapws.iapws1992 import psat

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', dest='path')
path = parser.parse_args().path


fig = pyplot.figure(figsize=(4, 4))
fig.set_tight_layout(True)

T0 = 273.15  # Temperature conversion offset


# Main plot ----------

# Data

ps = [pmelt_ice_Ih, pmelt_ice_III, pmelt_ice_V, pmelt_ice_VI, pmelt_ice_VII,
      psubl_ice_Ih, psat]

Ts = [numpy.linspace(251.165, 273.16, 50),
      numpy.linspace(251.165, 256.164, 25),
      numpy.linspace(256.164, 273.31, 25),
      numpy.linspace(273.31, 355.0, 25),
      numpy.linspace(355.0, 715.0, 100),
      numpy.linspace(50.0, 273.16, 100),
      numpy.linspace(273.16, 647.096, 100)]

# Phase coexistence lines
for p, T in zip(ps, Ts):
    pyplot.semilogy(T-T0, p(T)/1e6, 'k-', linewidth=1)

# Triple and critical points markers
for Tstar, pstar in \
    zip([273.16, 251.165, 256.164, 273.31, 355, 647.096], \
        [611.657e-6, 208.566, 350.1, 632.4, 2216, 22.064]):
    color = 'k' if Tstar == 647.096 else 'k'
    pyplot.plot([Tstar-T0], [pstar], 'ko', markersize=4, color=color)

# Triple point and critical point labels
pyplot.text(290-T0, 1e-4, 'Triple\npoint', fontsize=9)
pyplot.text(610-T0, 50, 'Critical\npoint', fontsize=9)

# Dashed lines separating ice phases
pyplot.plot([210-T0, 251.165-T0], [208.566]*2, 'k--')
pyplot.plot([215-T0, 256.164-T0], [800, 350.1], 'k--')
pyplot.plot([240-T0, 273.31-T0], [3000, 800], 'k--')
pyplot.plot([350-T0, 355-T0], [12000, 2216], 'k--')

# Ice phase labels
pyplot.text(220-T0, 2, 'Ih', fontsize=9)
pyplot.text(200-T0, 270, 'III', fontsize=9)
pyplot.text(230-T0, 800, 'V', fontsize=9)
pyplot.text(290-T0, 3000, 'VI', fontsize=9)
pyplot.text(415-T0, 6000, 'VII', fontsize=9)

# Main phase labels
pyplot.text(50, 2, 'Liquid', fontsize=9)
pyplot.text(450-T0, 2e-2, 'Vapour', fontsize=9)
pyplot.text(200-T0, 2e-2, 'Solid', fontsize=9)

# Standard pressure line
pyplot.plot([190-T0, 720-T0], [101.325e-3]*2, 'k:', linewidth=1)

# Axes
ax = pyplot.gca()

# x-axis ticks and labels
ax.set_yticks([10**n for n in range(-7, 5, 2)])
pyplot.xlabel(r'Temperature ($\mathregular{^{\circ}C}$)')

# y-axis ticks and labels
ax.set_yticks([10**n for n in range(-7, 5)])
ax.set_yticklabels(['', r'$\mathregular{1\ Pa}$', '', '',
                    r'$\mathregular{1\ kPa}$', '', '',
                    r'$\mathregular{1\ MPa}$', '', '',
                    r'$\mathregular{1\ GPa}$', ''])
pyplot.ylabel('Pressure')

# Axis limits
pyplot.xlim(190-T0, 720-T0)
pyplot.ylim(1.e-7, 2.e4)


# Inset plot ----------

# Data
ps = [pmelt_ice_Ih, psubl_ice_Ih, psat]

Ts = [numpy.linspace(273.12, 273.16, 100),
      numpy.linspace(273.12, 273.16, 100),
      numpy.linspace(273.16, 273.20, 100)]

# Axes
ax_inset = fig.add_axes([0.6, 0.25, 0.3, 0.2])

# Coexistence curves
for p, T in zip(ps, Ts):
    ax_inset.semilogy(T-T0, p(T)/1e6, 'k-', linewidth=1)

# Triple point
ax_inset.plot(273.16-T0, 611.657e-6, 'ko', markersize=4)

# Standard pressure line
ax_inset.plot([200.-T0, 720.-T0], [101.325e-3]*2, 'k:', linewidth=1)

# Solid and liquid markers
pyplot.text(273.135-T0, 5.e-3, 'S', fontsize=9)
pyplot.text(273.168-T0, 5.e-3, 'L', fontsize=9)

# x-axis ticks and labels
ax_inset.set_xticks([273.13-T0, 273.15-T0, 273.17-T0])
ax_inset.xaxis.set_minor_locator(ticker.FixedLocator([273.14-T0, 273.16-T0]))
ax_inset.set_xticklabels(['-0.02', '0', '0.02'], size=9)

# y-axis ticks and labels
ax_inset.set_yticks([10**n for n in range(-3, 0, 1)])
ax_inset.yaxis.set_minor_locator(ticker.FixedLocator([]))
ax_inset.set_yticklabels([])

# Axis limits
ax_inset.set_xlim(273.12-T0, 273.18-T0)
ax_inset.set_ylim(2.e-4, 1)

if path:
    pyplot.savefig(path)
else:
    pyplot.show()
