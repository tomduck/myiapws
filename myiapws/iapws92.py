# iapws92.py: Library for IAPWS 1992 saturation properties of water

# Copyright (C) 2014, 2016 Thomas J. Duck
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

"""IAPWS 1992 saturation properties of ordinary water substance.

Overview:

  This package implements the IAWPS 1992 formulas for thermodynamic
  properties on the coexistence curve of liquid water and vapour.  All
  functions have temperature (K) as the sole argument.

References:

  [1] Revised supplementary release on saturation properties of ordinary
      water substance.

      IAPWS, 1992.

      http://www.iapws.org/relguide/supsat.pdf
"""

# pylint: disable=invalid-name

import functools

import numpy


# Decorator to validate input
def _validate(f):
    """Decorator factory function to validate args."""

    @functools.wraps(f)
    def decorate(T):
        """Decorator to validates args."""
        if numpy.logical_or(T < 273.16, T > Tc).any():
            msg = '273.16 <= T <= 647.096 K for water vapour saturation line'
            raise ValueError(msg)
        ret = f(T)
        return numpy.asscalar(ret) if numpy.isscalar(ret) else ret

    decorate.__name__ = f.__name__
    return decorate


# PUBLIC API -----------------------------------------------------------------

# Critical point
Tc = 647.096   # K
pc = 22.064e6  # Pa
rhoc = 322.    # kg/m3

@_validate
def p(T):
    """Saturation vapour pressure (Pa)."""
    theta = T/Tc
    tau = 1.-theta
    return pc * numpy.exp(1./theta *
                          (_a[0]*tau+_a[1]*tau**1.5+_a[2]*tau**3+_a[3]*tau**3.5
                           +_a[4]*tau**4+_a[5]*tau**7.5))

@_validate
def dpdT(T):
    """Saturation line slope (Pa/K)."""
    theta = T/Tc
    tau = 1.-theta
    return p(T)*\
      (-Tc/T**2 * (_a[0]*tau+_a[1]*tau**1.5+_a[2]*tau**3+_a[3]*tau**3.5+\
                   _a[4]*tau**4+_a[5]*tau**7.5) - \
       1./T * (_a[0]+_a[1]*1.5*tau**0.5+_a[2]*3*tau**2+_a[3]*3.5*tau**2.5+\
               _a[4]*4*tau**3+_a[5]*7.5*tau**6.5))

@_validate
def rhop(T):
    """Density of the saturated liquid (kg/m3)."""
    theta = T/Tc
    tau = 1.-theta
    return rhoc*(1+_b[0]*tau**(1./3)+_b[1]*tau**(2./3)+_b[2]*tau**(5./3)+\
                 _b[3]*tau**(16./3)+_b[4]*tau**(43./3)+_b[5]*tau**(110./3))

@_validate
def rhopp(T):
    """Density of the saturated vapour (kg/m3)."""
    theta = T/Tc
    tau = 1.-theta
    return rhoc*numpy.exp(_c[0]*tau**(2./6)+_c[1]*tau**(4./6)+\
                          _c[2]*tau**(8./6)+_c[3]*tau**(18./6)+\
                          _c[4]*tau**(37./6)+_c[5]*tau**(71./6))

@_validate
def hp(T):
    """Specific enthalpy of the saturated liquid (J/kg)."""
    return _alpha(T) + T/rhop(T)*dpdT(T)

@_validate
def hpp(T):
    """Specific enthalpy of the saturated liquid (J/kg)."""
    return _alpha(T) + T/rhopp(T)*dpdT(T)

@_validate
def sp(T):
    """Specific entropy of the saturated liquid (J/kg/K)."""
    return _phi(T)+1/rhop(T)*dpdT(T)

@_validate
def spp(T):
    """Specific entropy of the saturated liquid (J/kg/K)."""
    return _phi(T)+1/rhopp(T)*dpdT(T)


# PRIVATE --------------------------------------------------------------------

# Coefficients
_a = numpy.array([-7.85951783, 1.84408259, -11.7866497, 22.6807411,
                  -15.9618719, 1.80122502])
_b = numpy.array([1.99274064, 1.09965342, -0.510839303, -1.75493479,
                  -45.5170352, -6.74694450e5])
_c = numpy.array([-2.03150240, -2.68302940, -5.38626492, -17.2991605,
                  -44.7586581, -63.9201063])
_d = numpy.array([-5.65134998e-8, 2690.66631, 127.287297, -135.003439,
                  0.981825814])
_d_alpha = -1135.905627715
_d_phi = 2319.5246
_alpha0 = 1000.
_phi0 = _alpha0/Tc


# Functions

def _alpha(T):
    """Auxiliary quantity for specific enthalpy."""
    theta = T/Tc
    return _alpha0*(_d_alpha + _d[0]*theta**-19+_d[1]*theta+_d[2]*theta**4.5+\
                    _d[3]*theta**5+_d[4]*theta**54.5)

def _phi(T):
    """Auxiliary quantity for specific entropy."""
    theta = T/Tc
    return _phi0*(_d_phi+19./20*_d[0]*theta**-20+_d[1]*numpy.log(theta)+\
                  9./7*_d[2]*theta**3.5+5./4*_d[3]*theta**4+\
                  109./107*_d[4]*theta**53.5)
