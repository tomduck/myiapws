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

"""Unit tests for myiapws modules."""

import unittest
import sys

import test_iapws1995
import test_iapws1992
import test_iapws2011
import test_iapws2006

_VERBOSE = False  # Flag for unit test verbosity

def run_tests():
    """Runs the unit tests."""

    Nerrors = 0
    Nfailures = 0

    for Test in [test_iapws1995.Test_public, test_iapws1995.Test_private,
                 test_iapws1992.Test_public, test_iapws1992.Test_private,
                 test_iapws2011.Test_public,
                 test_iapws2006.Test_public, test_iapws2006.Test_private]:
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
