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

"""Unit tests for myiapws.iapws2011 module."""

# All test values are from Table 1 in [1].

# References:
#
#   [1] Revised release on the pressure along the melting and sublimation
#       curves of ordinary water substance.
#
#       IAPWS, 2011.
#
#       http://www.iapws.org/relguide/MeltSub2011.pdf

# pylint: disable=invalid-name
# pylint: disable=wildcard-import, unused-wildcard-import

import unittest
import sys

import numpy

from myiapws.iapws2011 import *

_VERBOSE = False  # Flag for unit test verbosity

class Test_public(unittest.TestCase):
    """Tests public API."""

    def assert_close(self, v1, v2, rtol=1.e-8, atol=0.):
        """Asserts scalars or arrays v1 and v2 are close."""
        isclose = numpy.isclose(v1, v2, rtol=rtol, atol=atol)
        if numpy.isscalar(isclose):
            self.assert_(isclose)
        else:
            self.assert_(isclose.all())

    def test_all(self):
        """Test all of the functions."""
        self.assert_close(pmelt_ice_Ih(260), 138.268e6, rtol=1.e-6)
        self.assert_close(pmelt_ice_III(254), 268.685e6, rtol=1.e-5)
        self.assert_close(pmelt_ice_V(265), 479.640e6, rtol=1.e-6)
        self.assert_close(pmelt_ice_VI(320), 1356.76e6, rtol=1.e-5)
        self.assert_close(pmelt_ice_VII(550), 6308.71e6, rtol=1.e-6)
        self.assert_close(psubl_ice_Ih(230), 8.94735, rtol=1.e-6)


def run_tests():
    """Runs the unit tests."""
    Nerrors = 0
    Nfailures = 0

    for Test in [Test_public]:
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
