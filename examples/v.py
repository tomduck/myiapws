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

"""Specific volume of liquid water at normal pressure between 0 and 100 C.

First determine the densities (rho) that solve p(rho, T) = 101.325 kPa.  The
optimization is done using Newton's method.  Specific volume is 1/rho.
"""

# pylint: disable=invalid-name

import numpy
from scipy.optimize import newton
from matplotlib import pyplot

from myiapws import iapws1995, iapws1992


# Define a series of temperatures
T = numpy.linspace(273.2, 373.15, 200)

# Get the saturation densities
rho0 = iapws1992.rhosat_liquid(T)

# Get the density at normal pressure for this series of temperatures.
# Use the saturation densities as a first estimate.
# pylint: disable=cell-var-from-loop
rho = numpy.array([newton(lambda x: iapws1995.p(x, T_) - 101325, rho_)
                   for rho_, T_ in zip(rho0, T)])

v = 1/rho

# Ploting
fig = pyplot.figure(figsize=(5.5, 4))

ax = fig.add_axes([0.17, 0.15, 0.78, 0.8]) # main axes
ax.plot(T-273.15, v*1000, linewidth=2)
ax.set_xlim(0, 100)
ax.set_ylim(0.995, 1.050)
ax.set_xlabel(r'Temperature $\mathrm{(^\circ C)}$')
ax.set_ylabel(r'Specific volume $\mathrm{(L/kg)}$')

ax.annotate('', xy=(6, 1.002), xycoords='data',
            xytext=(20, 1.018), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

ax_inset = fig.add_axes([0.31, 0.58, 0.3, 0.3]) # Inset
ax_inset.plot(T[:100]-273.15, v[:100]*1000., linewidth=2)

ax_inset.set_xlim(0, 10)
ax_inset.set_ylim(1., 1.0003)
ax_inset.set_yticks([1., 1.0001, 1.0002, 1.0003])
ax_inset.set_yticklabels(['1.0000', '1.0001', '1.0002', '1.0003'])

pyplot.show()
