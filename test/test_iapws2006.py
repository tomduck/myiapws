#! /usr/bin/env python3

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

"""Unit tests for myiapws.iapws2006 module."""

# References:
#
#  [1] Revised Release on the Equation of State 2006 for H2O Ice Ih.
#
#      IAPWS, 2009.
#
#      http://iapws.org/relguide/Ice-Rev2009.pdf

# Test data from both references [1] are used.

# pylint: disable=invalid-name, protected-access
# pylint: disable=wildcard-import, unused-wildcard-import

import unittest
import sys

import numpy

from myiapws import iapws2006
from myiapws.iapws2006 import *

_VERBOSE = False  # Flag for unit test verbosity


class Test_public(unittest.TestCase):
    """Tests the public API."""

    def assert_close(self, v1, v2, rtol=1.e-8, atol=0.):
        """Asserts scalar or array values v1 and v2 are close."""
        isclose = numpy.isclose(v1, v2, rtol=rtol, atol=atol)
        if numpy.isscalar(isclose):
            self.assert_(isclose)
        else:
            self.assert_(isclose.all())

    def test_g(self):
        """Tests the Gibbs potential g(T, p)."""
        self.assert_close(g(Tt, pt), 0.611784135)
        self.assert_close(g(273.152519, 101325), 0.10134274069e3)
        self.assert_close(g(100, 100e6), -0.222296513088e6)
        self.assert_close(g([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [0.611784135, 0.10134274069e3, -0.222296513088e6])

    def test_h(self):
        """Tests the enthalpy h(T, p)."""
        self.assert_close(h(Tt, pt), -0.333444253966e6)
        self.assert_close(h(273.152519, 101325), -0.333354873637e6)
        self.assert_close(h(100, 100e6), -0.483491635676e6)
        self.assert_close(h([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [-0.333444253966e6, -0.333354873637e6,
                           -0.483491635676e6])

    def test_f(self):
        """Tests the Helmholtz potential f(T, p)."""
        self.assert_close(f(Tt, pt), -0.55446875e-1)
        self.assert_close(f(273.152519, 101325), -0.918701567e1)
        self.assert_close(f(100, 100e6), -0.328489902347e6)
        self.assert_close(f([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [-0.55446875e-1, -0.918701567e1, -0.328489902347e6])

    def test_u(self):
        """Tests the internal energy u(T, p)."""
        self.assert_close(u(Tt, pt), -0.333444921197e6)
        self.assert_close(u(273.152519, 101325), -0.333465403393e6)
        self.assert_close(u(100, 100e6), -0.589685024936e6)
        self.assert_close(u([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [-0.333444921197e6, -0.333465403393e6,
                           -0.589685024936e6])

    def test_s(self):
        """Tests the entropy s(T, p)."""
        self.assert_close(s(Tt, pt), -0.122069433940e4)
        self.assert_close(s(273.152519, 101325), -0.122076932550e4)
        self.assert_close(s(100, 100e6), -0.261195122589e4)
        self.assert_close(s([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [-0.122069433940e4, -0.122076932550e4,
                           -0.261195122589e4])

    def test_cp(self):
        """Tests the isobaric heat capacity cp(T, p)."""
        self.assert_close(cp(Tt, pt), 0.209678431622e4)
        self.assert_close(cp(273.152519, 101325), 0.209671391024e4)
        self.assert_close(cp(100, 100e6), 0.866333195517e3)
        self.assert_close(cp([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [0.209678431622e4, 0.209671391024e4,
                           0.866333195517e3])

    def test_rho(self):
        """Tests the density rho(T, p)."""
        self.assert_close(rho(Tt, pt), 0.916709492200e3)
        self.assert_close(rho(273.152519, 101325), 0.916721463419e3)
        self.assert_close(rho(100, 100e6), 0.941678203297e3)
        self.assert_close(rho([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [0.916709492200e3, 0.916721463419e3,
                           0.941678203297e3])

    def test_alpha(self):
        """Tests the volumetric thermal expansion coefficient alpha(T, p)."""
        self.assert_close(alpha(Tt, pt), 0.159863102566e-3)
        self.assert_close(alpha(273.152519, 101325), 0.159841589458e-3)
        self.assert_close(alpha(100, 100e6), 0.25849552820710e-4)
        self.assert_close(alpha([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [0.159863102566e-3, 0.159841589458e-3,
                           0.25849552820710e-4])

    def test_beta(self):
        """Tests the pressure coefficient beta(T, p)."""
        self.assert_close(beta(Tt, pt), 0.135714764659e7)
        self.assert_close(beta(273.152519, 101325), 0.135705899321e7)
        self.assert_close(beta(100, 100e6), 0.291466166994e6)
        self.assert_close(beta([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [0.135714764659e7, 0.135705899321e7,
                           0.291466166994e6])

    def test_kappa_T(self):
        """Tests the isothermal compressibility kappa_T(T, p)."""
        self.assert_close(kappa_T(Tt, pt), 0.117793449348e-9)
        self.assert_close(kappa_T(273.152519, 101325), 0.117785291765e-9)
        self.assert_close(kappa_T(100, 100e6), 0.886880048115e-10)
        self.assert_close(kappa_T([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [0.117793449348e-9, 0.117785291765e-9,
                           0.886880048115e-10])

    def test_kappa_S(self):
        """Tests the isentropic compressibility kappa_S(T, p)."""
        self.assert_close(kappa_S(Tt, pt), 0.114161597779e-9)
        self.assert_close(kappa_S(273.152519, 101325), 0.114154442556e-9)
        self.assert_close(kappa_S(100, 100e6), 0.886060982687e-10)
        self.assert_close(kappa_S([Tt, 273.152519, 100], [pt, 101325, 100e6]),
                          [0.114161597779e-9, 0.114154442556e-9,
                           0.886060982687e-10])


class Test_private(unittest.TestCase):
    """Tests the private API."""

    def assert_close(self, v1, v2, rtol=1.e-8, atol=0.):
        """Asserts scalar or array values v1 and v2 are close."""
        isclose = numpy.isclose(v1, v2, rtol=rtol, atol=atol)
        if numpy.isscalar(isclose):
            self.assert_(isclose)
        else:
            self.assert_(isclose.all())

    def test_gp(self):
        """Tests _gp()."""
        self.assert_close(iapws2006._gp(Tt, pt), 0.109085812737e-2)
        self.assert_close(iapws2006._gp(273.152519, 101325), 0.109084388214e-2)
        self.assert_close(iapws2006._gp(100, 100e6), 0.106193389260e-2)

    def test_gT(self):
        """Tests _gT()."""
        self.assert_close(iapws2006._gT(Tt, pt), 0.122069433940e4)
        self.assert_close(iapws2006._gT(273.152519, 101325), 0.122076932550e4)
        self.assert_close(iapws2006._gT(100, 100e6), 0.261195122589e4)

    def test_gpp(self):
        """Tests _gpp()."""
        self.assert_close(iapws2006._gpp(Tt, pt), -0.128495941571e-12)
        self.assert_close(iapws2006._gpp(273.152519, 101325),
                          -0.128485364928e-12)
        self.assert_close(iapws2006._gpp(100, 100e6), -0.941807981761e-13)

    def test_gTp(self):
        """Tests _gTp()."""
        self.assert_close(iapws2006._gTp(Tt, pt), 0.174387964700e-6)
        self.assert_close(iapws2006._gTp(273.152519, 101325), 0.174362219972e-6)
        self.assert_close(iapws2006._gTp(100, 100e6), 0.274505162488e-7)

    def test_gTT(self):
        """Tests _gTT()."""
        self.assert_close(iapws2006._gTT(Tt, pt), -0.767602985875e1)
        self.assert_close(iapws2006._gTT(273.152519, 101325), -0.767598233365e1)
        self.assert_close(iapws2006._gTT(100, 100e6), -0.866333195517e1)


def run_tests():
    """Runs unit tests on this module."""

    Nerrors = 0
    Nfailures = 0

    for Test in [Test_public, Test_private]:
        suite = unittest.makeSuite(Test)
        result = unittest.TextTestRunner(verbosity=2 if _VERBOSE else 1)\
          .run(suite)
        Nerrors += len(result.errors)
        Nfailures += len(result.failures)

    if Nerrors or Nfailures:
        print('\n\nSummary: %d errors and %d failures reported\n'%\
            (Nerrors, Nfailures))

    return Nerrors+Nfailures


if __name__ == '__main__':
    sys.exit(run_tests())

