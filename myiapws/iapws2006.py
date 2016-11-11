# iapws2006.py: Library for IAPWS 2006 thermodynamic properties of ice Ih

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

"""IAPWS 2006 thermodynamic properties of ice Ih.

Overview:

  This package implements the IAWPS 2006 formulas for thermodynamic
  properties of ice Ih.  Functions are of the form g(T, p) where T is
  the temperature (K) and p is the pressure (Pa).

References:

  [1] Revised Release on the Equation of State 2006 for H2O Ice Ih.

      IAPWS, 2009.

      http://iapws.org/relguide/Ice-Rev2009.pdf
"""

# pylint: disable=invalid-name

import numpy

from . import util


# PUBLIC API -----------------------------------------------------------------

# IAPWS 2006 constants (Table 1 in [1])

Tt = 273.16   # Triple-point temperature (K)
pt = 611.657  # Triple-point pressure (Pa)
p0 = 101325   # Normal pressure (Pa)


# IAPWS 2006 equations (Table 3 in [1])

@util.arrayfunc
def g(T, p):
    """Gibbs potential (J/kg)."""
    return _g(T, p)

@util.arrayfunc
def rho(T, p):
    """Density (kg/m^3)."""
    return 1/_gp(T, p)

@util.arrayfunc
def s(T, p):
    """Entropy (J/kg/K)."""
    return -_gT(T, p)

@util.arrayfunc
def cp(T, p):
    """Isobaric heat capacity (J/kg/K)."""
    return -T*_gTT(T, p)

@util.arrayfunc
def h(T, p):
    """Enthalpy (J/kg)."""
    return _g(T, p) - (T*_gT(T, p))

@util.arrayfunc
def u(T, p):
    """Internal energy (J/kg)."""
    return _g(T, p) - T*_gT(T, p) - p*_gp(T, p)

@util.arrayfunc
def f(T, p):
    """Helmholtz potential (J/kg)."""
    return _g(T, p) - p*_gp(T, p)

@util.arrayfunc
def alpha(T, p):
    """Volumetric thermal expansion coefficient (/K)."""
    return _gTp(T, p)/_gp(T, p)

@util.arrayfunc
def beta(T, p):
    """Pressure coefficient (Pa/K)."""
    return -_gTp(T, p)/_gpp(T, p)

@util.arrayfunc
def kappa_T(T, p):
    """Isothermal compressibility (/Pa)"""
    return -_gpp(T, p)/_gp(T, p)

@util.arrayfunc
def kappa_S(T, p):
    """Isentropic compressibility (/Pa)"""
    return (_gTp(T, p)**2-_gTT(T, p)*_gpp(T, p))/(_gp(T, p)*_gTT(T, p))


# PRIVATE --------------------------------------------------------------------

# IAPWS 2006 coefficients (Table 2 in [1]) ##

_g0k = numpy.array([-0.632020233335886e6, 0.655022213658955,
                    -0.189369929326131e-7, 0.339746123271053e-14,
                    -0.556464869058991e-21])

_s0 = 0.18913e3              # Absolute
_s0 = -0.332733756492168e4   # IAPWS95, expected by unit tests

_t = numpy.array([0.368017112855051e-1 + 0.510878114959572e-1j,
                  0.337315741065416 + 0.335449415919309j])

_t1 = _t[0]
_t2 = _t[1]

_r1 = 0.447050716285388e2 + 0.656876847463481e2j

_r2k = numpy.array([-0.725974574329220e2 - 0.781008427112870e2j,
                    -0.557107698030123e-4 + 0.464578634580806e-4j,
                    0.234801409215913e-10 - 0.285651142904972e-10j])


# IAPWS 2006 equations (Table 4 in [1])

# Gibbs potential and its derivatives

def _g(T, p):
    """Gibbs potential."""
    tau = T/Tt
    return _g0(p) - _s0*Tt*tau + Tt*numpy.real(
        util.sum(_r(p)*((_t-tau)*numpy.log(_t-tau)+(_t+tau)*numpy.log(_t+tau)
                        -2*_t*numpy.log(_t) - tau**2/_t)))

def _gT(T, p):
    """Partial derivative of g wrt T."""
    tau = T/Tt
    return -_s0 + numpy.real(util.sum(
        _r(p)*(-numpy.log(_t-tau) + numpy.log(_t+tau) - 2*tau/_t)))

def _gp(T, p):
    """Partial derivative of g wrt p."""
    tau = T/Tt
    return _g0p(p) + Tt*numpy.real(
        _r2p(p)*((_t2-tau)*numpy.log(_t2-tau) +
                 (_t2+tau)*numpy.log(_t2+tau) - 2*_t2*numpy.log(_t2) -
                 tau**2/_t2))

def _gTT(T, p):
    """Second partial derivative of g wrt T."""
    tau = T/Tt
    return 1/Tt*numpy.real(util.sum(_r(p)*(1/(_t-tau) + 1/(_t+tau) - 2/_t)))

def _gTp(T, p):
    """Second partial derivative of g wrt T and p."""
    tau = T/Tt
    return numpy.real(_r2p(p)*(-numpy.log(_t2-tau) +
                               numpy.log(_t2+tau) - 2*tau/_t2))

def _gpp(T, p):
    """Second partial derivative of g wrt p."""
    tau = T/Tt
    return _g0pp(p) + Tt*numpy.real(
        _r2pp*((_t2-tau)*numpy.log(_t2-tau) +
               (_t2+tau)*numpy.log(_t2+tau) -
               2*_t2*numpy.log(_t2) - tau**2/_t2))


# g0 and its derivatives

def _g0(p):
    """Coefficient g0 from Table 2 in [1]."""
    pi = p/pt
    pi0 = p0/pt
    k = numpy.arange(5)
    return util.sum(_g0k*(pi-pi0)**k)

def _g0p(p):
    """Partial derivative of g0 wrt p."""
    pi = p/pt
    pi0 = p0/pt
    k = numpy.arange(1, 5)
    return util.sum(_g0k[1:]*k/pt*(pi-pi0)**(k-1))

def _g0pp(p):
    """Second partial derivative of g0 wrt p."""
    pi = p/pt
    pi0 = p0/pt
    k = numpy.arange(2, 5)
    return util.sum(_g0k[2:]*k*(k-1)/pt**2*(pi-pi0)**(k-2))


# r2 and its derivatives

def _r2(p):
    """Coefficient r2 from Table 2 in [1]."""
    pi = p/pt
    pi0 = p0/pt
    k = numpy.arange(3)
    return util.sum(_r2k*(pi-pi0)**k)

def _r2p(p):
    """Partial derivative of r2 wrt p."""
    pi = p/pt
    pi0 = p0/pt
    k = numpy.arange(1, 3)
    return util.sum(_r2k[1:]*k/pt*(pi-pi0)**(k-1))

_r2pp = _r2k[2]*2/pt**2  # Second partial derivative of r2 wrt p


# Concatenated arrays

def _r(p):
    """Coefficient r from Table 2 in [1]."""
    r2 = _r2(p)
    return numpy.array([numpy.zeros_like(r2) + _r1, r2]).T.squeeze()
