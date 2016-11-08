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

"""Unit tests for myiapws.iapws92 module."""

# All test values are from Table 1 in [1].

# References:
#
#   [1] Revised supplementary release on saturation properties of ordinary
#       water substance.
#
#       IAPWS, 1992.
#
#       http://www.iapws.org/relguide/supsat.pdf

# pylint: disable=invalid-name, protected-access
# pylint: disable=wildcard-import, unused-wildcard-import

import sys
import unittest

import numpy

from myiapws import iapws92
from myiapws.iapws92 import *

_VERBOSE = False  # Flag for unit test verbosity


class Test_public(unittest.TestCase):
    """Tests public API."""

    T1 = 273.16
    T2 = 373.1243
    T3 = 647.096
    T = numpy.array([273.16, 373.1243, 647.096])

    def assert_close(self, v1, v2, rtol=1.e-8, atol=0.):
        """Asserts scalars or arrays v1 and v2 are close."""
        isclose = numpy.isclose(v1, v2, rtol=rtol, atol=atol)
        if numpy.isscalar(isclose):
            self.assert_(isclose)
        else:
            self.assert_(isclose.all())

    def test_psat(self):
        """Tests the saturation pressure."""

        self.assert_close(psat(self.T1), 611.657, rtol=1.e-6)
        self.assert_close(psat(self.T2), 0.101325e6, rtol=1.e-6)
        self.assert_close(psat(self.T3), 22.064e6, rtol=1.e-5)
        self.assert_close(psat(self.T), [611.657, 0.101325e6, 22.064e6],
                          rtol=1.e-5)

        # Make sure masked values don't cause an error
        self.assertEqual(psat(numpy.ma.masked_values(Tc, Tc)).mask, True)

    def test_dpdT(self):
        """Tests the saturation line slope."""
        self.assert_close(dpdT(self.T1), 44.436693, rtol=1.e-8)
        self.assert_close(dpdT(self.T2), 3.616e3, rtol=2.e-4)
        self.assert_close(dpdT(self.T3), 268.e3, rtol=1.e-3)
        self.assert_close(dpdT(self.T), [44.436693, 3.616e3, 268.e3],
                          rtol=1.e-3)

    def test_rhosat_liquid(self):
        """Tests the saturation liquid density."""
        self.assert_close(rhosat_liquid(self.T1), 999.789, rtol=1.e-6)
        self.assert_close(rhosat_liquid(self.T2), 958.365, rtol=1.e-6)
        self.assert_close(rhosat_liquid(self.T3), 322., rtol=1.e-3)
        self.assert_close(rhosat_liquid(self.T), [999.7897, 958.365, 322.],
                          rtol=1.e-3)

    def test_rhosat_vapor(self):
        """Tests the saturation vapor density."""
        self.assert_close(rhosat_vapor(self.T1), 0.00485426, rtol=1.e-6)
        self.assert_close(rhosat_vapor(self.T2), 0.597586, rtol=1.e-6)
        self.assert_close(rhosat_vapor(self.T3), 322., rtol=1.e-3)
        self.assert_close(rhosat_vapor(self.T), [0.00485426, 0.597586, 322.],
                          rtol=1.e-3)

    def test_hsat_liquid(self):
        """Tests the saturation liquid enthalpy."""
        self.assert_close(hsat_liquid(self.T1), 0.611786, rtol=1.e-6)
        self.assert_close(hsat_liquid(self.T2), 419.05e3, rtol=2.e-5)
        self.assert_close(hsat_liquid(self.T3), 2086.6e3, rtol=2.e-5)
        self.assert_close(hsat_liquid(self.T), [0.611786, 419.05e3, 2086.6e3],
                          rtol=2.e-5)

    def test_hsat_vapor(self):
        """Tests the saturation vapor enthalpy."""
        self.assert_close(hsat_vapor(self.T1), 2500.5e3, rtol=2.e-5)
        self.assert_close(hsat_vapor(self.T2), 2675.7e3, rtol=1.e-5)
        self.assert_close(hsat_vapor(self.T3), 2086.6e3, rtol=2.e-5)
        self.assert_close(hsat_vapor(self.T), [2500.5e3, 2675.7e3, 2086.6e3],
                          rtol=2.e-5)

    def test_ssat_liquid(self):
        """Tests the saturation liquid entropy."""
        self.assert_close(ssat_liquid(self.T1), 0., atol=1.e-5)
        self.assert_close(ssat_liquid(self.T2), 1.307e3, rtol=1.e-4)
        self.assert_close(ssat_liquid(self.T3), 4.410e3, rtol=1.e-4)
        self.assert_close(ssat_liquid(self.T), [0., 1.307e3, 4.410e3],
                          rtol=1.e-4, atol=1.e-5)

    def test_ssat_vapor(self):
        """Tests the saturation vapor entropy."""
        self.assert_close(ssat_vapor(self.T1), 9.154e3, rtol=1.e-4)
        self.assert_close(ssat_vapor(self.T2), 7.355e3, rtol=1.e-4)
        self.assert_close(ssat_vapor(self.T3), 4.410e3, rtol=1.e-4)
        self.assert_close(ssat_vapor(self.T), [9.154e3, 7.355e3, 4.410e3],
                          rtol=1.e-4)


class Test_private(unittest.TestCase):
    """Tests private functions."""

    T1 = 273.16
    T2 = 373.1243
    T3 = 647.096

    def assert_close(self, v1, v2, rtol=1.e-8, atol=0.):
        """Asserts scalars or arrays v1 and v2 are close."""
        isclose = numpy.isclose(v1, v2, rtol=rtol, atol=atol)
        if numpy.isscalar(isclose):
            self.assert_(isclose)
        else:
            self.assert_(isclose.all())

    def test_alpha(self):
        """Tests auxiliary quantity for specific enthalpy."""
        self.assert_close(iapws92._alpha(self.T1), -11.529101)
        self.assert_close(iapws92._alpha(self.T2), 417.65e3, rtol=1.e-5)
        self.assert_close(iapws92._alpha(self.T3), 1548e3, rtol=1.e-4)

    def test_phi(self):
        """Tests auxiliary quantity for specific entropy."""
        self.assert_close(iapws92._phi(self.T1), -0.04, rtol=2.e-1)
        self.assert_close(iapws92._phi(self.T2), 1.303e3, rtol=2.e-4)
        self.assert_close(iapws92._phi(self.T3), 3.578e3, rtol=1.e-4)


def run_tests():
    """Runs the unit tests."""
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
