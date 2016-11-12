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

"""Unit tests for myiapws.iapws1995 module."""

# References:
#
#   [1] Revised release on the IAPWS formulation 1995 for the thermodynamic
#       properties of ordinary water substance for general and scientific use
#
#       IAPWS, 2014.
#
#       http://iapws.org/relguide/IAPWS95-2014.pdf
#
#   [2] The IAPWS formulation 1995 for the thermodynamic properties of
#       ordinary water substance for general and scientific use
#
#        Wagner, W., and A. Pruss, J. Phys. Chem. Ref. Data, 31, 387-535, 2002.
#
#       http://www.nist.gov/data/PDFfiles/jpcrd617.pdf

# Test data from both references [1] and [2] are used.  For quantities where
# test data is not given, thermodynamic identities are checked.

# pylint: disable=invalid-name, protected-access
# pylint: disable=wildcard-import, unused-wildcard-import

import sys
import unittest

import numpy

from myiapws import iapws1995
from myiapws.iapws1995 import *

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


    def test_f(self):
        """Tests the Hemlholtz potential f(rho, T)."""
        # test_u() performs additional consistency checks
        self.assert_close(f(rhoc, Tc), fc)


    def test_p(self):
        """Tests the vapour pressure, p(rho, T), in the single-phase region.
        See Table 7 in [1]."""

        self.assert_close(p(0.9965560e3, 300)/1.e6, 0.992418352e-1)
        self.assert_close(p(0.1005308e4, 300)/1.e6, 0.200022515e2)
        self.assert_close(p(0.1188202e4, 300)/1.e6, 0.700004704e3)

        self.assert_close(p(0.4350000, 500)/1.e6, 0.999679423e-1)
        self.assert_close(p(0.4532000e1, 500)/1.e6, 0.999938125)
        self.assert_close(p(0.8380250e3, 500)/1.e6, 0.100003858e2)
        self.assert_close(p(0.1084564e4, 500)/1.e6, 0.700000405e3)

        self.assert_close(p(0.3580000e3, 647)/1.e6, 0.220384756e2)

        self.assert_close(p(0.2410000, 900)/1.e6, 0.100062559)
        self.assert_close(p(0.5261500e2, 900)/1.e6, 0.200000690e2)
        self.assert_close(p(0.8707690e3, 900)/1.e6, 0.700000006e3)

        self.assert_close(p(rhoc, Tc), pc)

        self.assert_close(
            p([0.9965560e3, 0.1005308e4, 0.1188202e4], [300]*3)/1.e6,
            [0.992418352e-1, 0.200022515e2, 0.700004704e3])

        self.assert_close(
            p([0.9965560e3, 0.1005308e4, 0.1188202e4, rhoc], [300]*3+[Tc])/1.e6,
            [0.992418352e-1, 0.200022515e2, 0.700004704e3, pc/1.e6])


    def test_u(self):
        """Tests the internal energy, u(rho, T)."""

        def get_u(rho, T):
            """Returns u = f + Ts."""
            return f(rho, T)+T*s(rho, T)

        # Test u from formulation against the identity u = f + Ts

        self.assert_close(u(0.9965560e3, 300), get_u(0.9965560e3, 300))
        self.assert_close(u(0.1005308e4, 300), get_u(0.1005308e4, 300))
        self.assert_close(u(0.1188202e4, 300), get_u(0.1188202e4, 300))

        self.assert_close(u(0.4350000, 500), get_u(0.4350000, 500))
        self.assert_close(u(0.4532000e1, 500), get_u(0.4532000e1, 500))
        self.assert_close(u(0.8380250e3, 500), get_u(0.8380250e3, 500))
        self.assert_close(u(0.1084564e4, 500), get_u(0.1084564e4, 500))

        self.assert_close(u(0.3580000e3, 647), get_u(0.3580000e3, 647))

        self.assert_close(u(0.2410000, 900), get_u(0.2410000, 900))
        self.assert_close(u(0.5261500e2, 900), get_u(0.5261500e2, 900))
        self.assert_close(u(0.8707690e3, 900), get_u(0.8707690e3, 900))

        self.assert_close(u(rhoc, Tc), uc)

        self.assert_close(
            u([0.9965560e3, 0.1005308e4, 0.1188202e4], [300.]*3),
            [get_u(0.9965560e3, 300), get_u(0.1005308e4, 300),
             get_u(0.1188202e4, 300)])

        self.assert_close(
            u([0.9965560e3, 0.1005308e4, 0.1188202e4, rhoc], [300.]*3+[Tc]),
            [get_u(0.9965560e3, 300), get_u(0.1005308e4, 300),
             get_u(0.1188202e4, 300), uc])


    def test_s(self):
        """Tests the specific entropy, s(rho, p), in the single-phase region.
        See Table 7 in [1]."""

        self.assert_close(s(0.9965560e3, 300)/1.e3, 0.393062643)
        self.assert_close(s(0.1005308e4, 300)/1.e3, 0.387405401)
        self.assert_close(s(0.1188202e4, 300)/1.e3, 0.132609616)

        self.assert_close(s(0.4350000, 500)/1.e3, 0.794488271e1)
        self.assert_close(s(0.4532000e1, 500)/1.e3, 0.682502725e1)
        self.assert_close(s(0.8380250e3, 500)/1.e3, 0.256690919e1)
        self.assert_close(s(0.1084564e4, 500)/1.e3, 0.203237509e1)

        self.assert_close(s(0.3580000e3, 647)/1.e3, 0.432092307e1)

        self.assert_close(s(0.2410000, 900)/1.e3, 0.916653194e1)
        self.assert_close(s(0.5261500e2, 900)/1.e3, 0.659070225e1)
        self.assert_close(s(0.8707690e3, 900)/1.e3, 0.417223802e1)

        self.assert_close(s(rhoc, Tc), sc)

        self.assert_close(
            s([0.9965560e3, 0.1005308e4, 0.1188202e4], [300]*3)/1.e3,
            [0.393062643, 0.387405401, 0.132609616])

        self.assert_close(
            s([0.9965560e3, 0.1005308e4, 0.1188202e4, rhoc], [300]*3+[Tc])/1.e3,
            [0.393062643, 0.387405401, 0.132609616, sc/1.e3])


    def test_h(self):  # h = u+pv
        """Tests the enthalpy, h(rho, T)."""

        def get_h(rho, T):
            """Returns h = u + p/rho."""
            return u(rho, T)+p(rho, T)/rho

        # Test h from formulation against the identity h = u + p/rho

        self.assert_close(h(0.9965560e3, 300), get_h(0.9965560e3, 300))

        self.assert_close(h(rhoc, Tc), hc)


    def test_cv(self):
        """Tests the isochoric specific heat capacity, cv(rho, p), in the
        single-phase region.  See Table 7 in [1]."""

        self.assert_close(cv(0.9965560e3, 300)/1.e3, 0.413018112e1)
        self.assert_close(cv(0.1005308e4, 300)/1.e3, 0.406798347e1)
        self.assert_close(cv(0.1188202e4, 300)/1.e3, 0.346135580e1)

        self.assert_close(cv(0.4350000, 500)/1.e3, 0.150817541e1)
        self.assert_close(cv(0.4532000e1, 500)/1.e3, 0.166991025e1)
        self.assert_close(cv(0.8380250e3, 500)/1.e3, 0.322106219e1)
        self.assert_close(cv(0.1084564e4, 500)/1.e3, 0.307437693e1)

        self.assert_close(cv(0.3580000e3, 647)/1.e3, 0.618315728e1)

        self.assert_close(cv(0.2410000, 900)/1.e3, 0.175890657e1)
        self.assert_close(cv(0.5261500e2, 900)/1.e3, 0.193510526e1)
        self.assert_close(cv(0.8707690e3, 900)/1.e3, 0.266422350e1)

        self.assertEqual(cv(rhoc, Tc), cvc)

        self.assert_close(
            cv([0.9965560e3, 0.1005308e4, 0.1188202e4], [300]*3)/1.e3,
            [0.413018112e1, 0.406798347e1, 0.346135580e1])

        self.assert_close(
            cv([0.9965560e3, 0.1005308e4, 0.1188202e4, rhoc],
               [300]*3+[Tc])/1.e3,
            [0.413018112e1, 0.406798347e1, 0.346135580e1, cvc/1.e3])


    def test_cp(self):
        """Tests the isobaric heat capacity, cp(rho, p)."""

        # Value from Table 13.2 in [2], pg. 496
        self.assert_close(cp(0.9965560e3, 300), 4180.6, rtol=1.e-5)

        # Critical value
        self.assertEqual(cp(rhoc, Tc), cpc)


    def test_w(self):
        """Tests the speed of sound, w(rho, p), in the single-phase region.
        See Table 7 in [1]."""

        self.assert_close(w(0.9965560e3, 300), 0.150151914e4)
        self.assert_close(w(0.1005308e4, 300), 0.153492501e4)
        self.assert_close(w(0.1188202e4, 300), 0.244357992e4)

        self.assert_close(w(0.4350000, 500), 0.548314253e3)
        self.assert_close(w(0.4532000e1, 500), 0.535739001e3)
        self.assert_close(w(0.8380250e3, 500), 0.127128441e4)
        self.assert_close(w(0.1084564e4, 500), 0.241200877e4)

        self.assert_close(w(0.3580000e3, 647), 0.252145078e3)

        self.assert_close(w(0.2410000, 900), 0.724027147e3)
        self.assert_close(w(0.5261500e2, 900), 0.698445674e3)
        self.assert_close(w(0.8707690e3, 900), 0.201933608e4)

        self.assertEqual(w(rhoc, Tc), wc)

        self.assert_close(
            w([0.9965560e3, 0.1005308e4, 0.1188202e4], [300]*3),
            [0.150151914e4, 0.153492501e4, 0.244357992e4])

        self.assert_close(
            w([0.9965560e3, 0.1005308e4, 0.1188202e4, rhoc], [300]*3+[Tc]),
            [0.150151914e4, 0.153492501e4, 0.244357992e4, wc])


    def test_mu(self):
        """Tests the Joule-Thomson coefficient, mu(rho, T)."""

        def get_mu(rho, T):
            """Returns mu = sqrt(T(cp/cv)(cp-cv))/(w rho cp)-1/(rho cp)."""
            cp_, cv_ = cp(rho, T), cv(rho, T)
            return numpy.sqrt(T*(cp_/cv_)*(cp_-cv_))/(w(rho, T)*rho*cp_)-\
              1/(rho*cp_)

        # Test mu from formulation against the identity
        # mu = sqrt(T(cp/cv)(cp-cv))/(w rho cp)-1/(rho cp)

        self.assert_close(mu(0.9965560e3, 300), get_mu(0.9965560e3, 300))
        self.assert_close(mu(0.1005308e4, 300), get_mu(0.1005308e4, 300))
        self.assert_close(mu(0.1188202e4, 300), get_mu(0.1188202e4, 300))

        self.assert_close(mu(0.4350000, 500), get_mu(0.4350000, 500))
        self.assert_close(mu(0.4532000e1, 500), get_mu(0.4532000e1, 500))
        self.assert_close(mu(0.8380250e3, 500), get_mu(0.8380250e3, 500))
        self.assert_close(mu(0.1084564e4, 500), get_mu(0.1084564e4, 500))

        self.assert_close(mu(0.3580000e3, 647), get_mu(0.3580000e3, 647))

        self.assertRaises(NotImplementedError, mu, rhoc, Tc)

        self.assert_close(mu(0.2410000, 900), get_mu(0.2410000, 900))
        self.assert_close(mu(0.5261500e2, 900), get_mu(0.5261500e2, 900))
        self.assert_close(mu(0.8707690e3, 900), get_mu(0.8707690e3, 900))

        self.assert_close(
            mu([0.9965560e3, 0.1005308e4, 0.1188202e4], [300]*3),
            [get_mu(0.9965560e3, 300), get_mu(0.1005308e4, 300),
             get_mu(0.1188202e4, 300)])


    def test_deltaT(self):
        """Tests the isothermal throttling coefficient, deltaT(rho, T)."""

        dT = deltaT

        def get_dT(rho, T):
            """Returns dT = - mu cp."""
            # Ref. [2] eq. 4.2, pg. 416
            return -mu(rho, T)*cp(rho, T)

        # Test deltaT from formulation against the identity dT = - mu cp

        self.assert_close(dT(0.9965560e3, 300), get_dT(0.9965560e3, 300))
        self.assert_close(dT(0.1005308e4, 300), get_dT(0.1005308e4, 300))
        self.assert_close(dT(0.1188202e4, 300), get_dT(0.1188202e4, 300))

        self.assert_close(dT(0.4350000, 500), get_dT(0.4350000, 500))
        self.assert_close(dT(0.4532000e1, 500), get_dT(0.4532000e1, 500))
        self.assert_close(dT(0.8380250e3, 500), get_dT(0.8380250e3, 500))
        self.assert_close(dT(0.1084564e4, 500), get_dT(0.1084564e4, 500))

        self.assert_close(dT(0.3580000e3, 647), get_dT(0.3580000e3, 647))

        self.assert_close(dT(0.2410000, 900), get_dT(0.2410000, 900))
        self.assert_close(dT(0.5261500e2, 900), get_dT(0.5261500e2, 900))
        self.assert_close(dT(0.8707690e3, 900), get_dT(0.8707690e3, 900))

        self.assertRaises(NotImplementedError, dT, rhoc, Tc)

        self.assert_close(
            dT([0.9965560e3, 0.1005308e4, 0.1188202e4], [300]*3),
            [get_dT(0.9965560e3, 300), get_dT(0.1005308e4, 300),
             get_dT(0.1188202e4, 300)])


    def test_betas(self):
        """Tests the isentropic temperature-pressure coefficient,
        betas(rho, T)."""

        Bs = betas

        def get_Bs(rho, T):
            """Returns Bs = T v alpha/cp."""
            # From reduction of derivatives
            return T*alpha(rho, T)/(cp(rho, T)*rho)

        # Test betas from formulation against the identity
        # betas = T v alpha/cp

        self.assert_close(Bs(0.9965560e3, 300), get_Bs(0.9965560e3, 300))
        self.assert_close(Bs(0.1005308e4, 300), get_Bs(0.1005308e4, 300))
        self.assert_close(Bs(0.1188202e4, 300), get_Bs(0.1188202e4, 300))

        self.assert_close(Bs(0.4350000, 500), get_Bs(0.4350000, 500))
        self.assert_close(Bs(0.4532000e1, 500), get_Bs(0.4532000e1, 500))
        self.assert_close(Bs(0.8380250e3, 500), get_Bs(0.8380250e3, 500))
        self.assert_close(Bs(0.1084564e4, 500), get_Bs(0.1084564e4, 500))

        self.assert_close(Bs(0.3580000e3, 647), get_Bs(0.3580000e3, 647))

        self.assert_close(Bs(0.2410000, 900), get_Bs(0.2410000, 900))
        self.assert_close(Bs(0.5261500e2, 900), get_Bs(0.5261500e2, 900))
        self.assert_close(Bs(0.8707690e3, 900), get_Bs(0.8707690e3, 900))

        self.assertRaises(NotImplementedError, Bs, rhoc, Tc)

        self.assert_close(
            Bs([0.9965560e3, 0.1005308e4, 0.1188202e4], [300]*3),
            [get_Bs(0.9965560e3, 300), get_Bs(0.1005308e4, 300),
             get_Bs(0.1188202e4, 300)])


    def test_B(self):
        """Tests second virial coefficient, B(T)."""

        rho = iapws1995._DELTAMIN*rhoc

        def get_mu(rho, T):
            """Returns (T * dB/dT - BT)/cp."""
            # Ref. [2] eq. 4.1 (pg. 416)
            dBdT = (B(T+0.00005)-B(T-0.00005))/0.0001  # Two-point derivative
            return (T*dBdT-B(T))/cp(rho, T)

        # Test B from formulation against the identity
        # B = (T * dB/dT - BT)/cp

        self.assert_close(mu(rho, Tt), get_mu(rho, Tt))
        self.assert_close(mu(rho, 300), get_mu(rho, 300))
        self.assert_close(mu(rho, 400), get_mu(rho, 400))
        self.assert_close(mu(rho, 500), get_mu(rho, 500))
        self.assert_close(mu(rho, 600), get_mu(rho, 600))
        self.assert_close(mu(rho, Tc), get_mu(rho, Tc))
        self.assert_close(mu(rho, 700), get_mu(rho, 700))
        self.assert_close(mu(rho, 800), get_mu(rho, 800))
        self.assert_close(mu(rho, 900), get_mu(rho, 900))
        self.assert_close(mu(rho, 1000), get_mu(rho, 1000))


    def test_C(self):
        """Tests third virial coefficient, C(T)."""
        pass


    # pylint: disable=too-many-statements
    def test_rhosat(self):
        """Tests the saturation density, rhosat(T)."""

        # Values from Table 13.1 in [2], pg. 486

        T = 250.
        self.assertRaises(ValueError, rhosat, T)

        T = Tt
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 999.793, rtol=1.e-6)
        self.assert_close(rhov, 0.00485, rtol=1.e-3)
        self.assert_close(p(rhol, T), 612, rtol=1.e-3)
        self.assert_close(p(rhov, T), 612, rtol=1.e-3)

        T = 300
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 996.513, rtol=1.e-6)
        self.assert_close(rhov, 0.02559, rtol=1.e-4)
        self.assert_close(p(rhol, T), 3537, rtol=1.e-4)
        self.assert_close(p(rhov, T), 3537, rtol=1.e-4)

        T = 350
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 973.702, rtol=1.e-6)
        self.assert_close(rhov, 0.26029, rtol=1.e-5)
        self.assert_close(p(rhol, T), 41682, rtol=1.e-5)
        self.assert_close(p(rhov, T), 41682, rtol=1.e-5)

        T = 400
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 937.486, rtol=1.e-6)
        self.assert_close(rhov, 1.3694, rtol=1.e-5)
        self.assert_close(p(rhol, T), 245770, rtol=1.e-5)
        self.assert_close(p(rhov, T), 245770, rtol=1.e-5)

        T = 450
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 890.341, rtol=1.e-6)
        self.assert_close(rhov, 4.8120, rtol=1.e-5)
        self.assert_close(p(rhol, T), 932200, rtol=1.e-5)
        self.assert_close(p(rhov, T), 932200, rtol=1.e-5)

        T = 500
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 831.313, rtol=1.e-6)
        self.assert_close(rhov, 13.199, rtol=1.e-5)
        self.assert_close(p(rhol, T), 2.6392e6, rtol=1.e-5)
        self.assert_close(p(rhov, T), 2.6392e6, rtol=1.e-5)

        T = 550
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 755.808, rtol=1.e-6)
        self.assert_close(rhov, 31.474, rtol=1.e-5)
        self.assert_close(p(rhol, T), 6.1172e6, rtol=1.e-5)
        self.assert_close(p(rhov, T), 6.1172e6, rtol=1.e-5)

        T = 600
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 649.411, rtol=1.e-6)
        self.assert_close(rhov, 72.842, rtol=1.e-5)
        self.assert_close(p(rhol, T), 12.345e6, rtol=2.e-5)
        self.assert_close(p(rhov, T), 12.345e6, rtol=2.e-5)

        T = 620
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 586.88, rtol=1.e-5)
        self.assert_close(rhov, 106.31, rtol=5.e-5)
        self.assert_close(p(rhol, T), 15.901e6, rtol=5.e-5)
        self.assert_close(p(rhov, T), 15.901e6, rtol=5.e-5)

        T = 640
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 481.53, rtol=1.e-5)
        self.assert_close(rhov, 177.15, rtol=5.e-5)
        self.assert_close(p(rhol, T), 20.265e6, rtol=5.e-5)
        self.assert_close(p(rhov, T), 20.265e6, rtol=5.e-5)

        T = 645
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 425.05, rtol=1.e-5)
        self.assert_close(rhov, 224.45, rtol=5.e-5)
        self.assert_close(p(rhol, T), 21.515e6, rtol=5.e-5)
        self.assert_close(p(rhov, T), 21.515e6, rtol=5.e-5)

        T = 646
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 402.96, rtol=1.e-5)
        self.assert_close(rhov, 243.46, rtol=1.e-5)
        self.assert_close(p(rhol, T), 21.775e6, rtol=1.e-5)
        self.assert_close(p(rhov, T), 21.775e6, rtol=1.e-5)

        T = 647
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, 357.34, rtol=1.e-5)
        self.assert_close(rhov, 286.51, rtol=1.e-5)
        self.assert_close(p(rhol, T), 22.038e6, rtol=2.e-5)
        self.assert_close(p(rhov, T), 22.038e6, rtol=2.e-5)

        T = Tc
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, rhoc, rtol=1.e-3)
        self.assert_close(rhov, rhoc, rtol=1.e-3)
        self.assert_close(p(rhol, T), p(rhov, T), rtol=1.e-6)

        # This is maximum T for success (except Tc itself).  Numerical errors
        # occur if we try to go closer to the critical point.
        T = 647.0959
        rhol, rhov = rhosat(T)

        T = 650.
        self.assertRaises(ValueError, rhosat, T)


        T = [300, 350, 400, 450, 500, 550, 600, 620, 640, 645, 646, 647]
        rhol, rhov = rhosat(T)
        self.assert_close(rhol, [996.513, 973.702, 937.486, 890.341, 831.313,
                                 755.808, 649.411, 586.88, 481.53, 425.05,
                                 402.96, 357.34], rtol=1.e-5)
        self.assert_close(rhov, [0.02559, 0.26029, 1.3694, 4.8120, 13.199,
                                 31.474, 72.842, 106.31, 177.15, 224.45, 243.46,
                                 286.51], rtol=1.e-4)


class Test_private(unittest.TestCase):
    """Tests the private API."""

    # See Table 6 in [1] and Table 6.6 in [2] (pg. 436).

    def assert_close(self, v1, v2, rtol=1.e-8, atol=0.):
        """Asserts scalar values v1 and v2 are close."""
        isclose = numpy.isclose(v1, v2, rtol=rtol, atol=atol)
        self.assert_(isclose)

    def test_arrays(self):
        """Tests array lengths."""
        self.assertEqual(len(iapws1995._no), 8)
        self.assertEqual(len(iapws1995._gammao), 8)
        self.assertEqual(len(iapws1995._c), 51)
        self.assertEqual(len(iapws1995._d), 54)
        self.assertEqual(len(iapws1995._t), 54)
        self.assertEqual(len(iapws1995._n), 56)
        self.assertEqual(len(iapws1995._a), 2)
        self.assertEqual(len(iapws1995._b), 2)
        self.assertEqual(len(iapws1995._B), 2)
        self.assertEqual(len(iapws1995._alpha), 3)
        self.assertEqual(len(iapws1995._beta1), 3)
        self.assertEqual(len(iapws1995._gamma), 3)
        self.assertEqual(len(iapws1995._epsilon), 3)
        self.assertEqual(len(iapws1995._C), 2)
        self.assertEqual(len(iapws1995._D), 2)
        self.assertEqual(len(iapws1995._A), 2)
        self.assertEqual(len(iapws1995._beta2), 2)

    def test_phio(self):
        """Tests _phio()."""
        self.assert_close(iapws1995._phio(838.025/rhoc, Tc/500), 0.204797733e1)
        self.assert_close(iapws1995._phio(358/rhoc, Tc/647), -0.156319605e1)

    def test_phio_delta(self):
        """Tests _phio_delta()."""
        self.assert_close(iapws1995._phio_delta(838.025/rhoc), 0.384236747)
        self.assert_close(iapws1995._phio_delta(358/rhoc), 0.899441341)

    def test_phio_deltadelta(self):
        """Tests _phio_deltadelta()."""
        self.assert_close(iapws1995._phio_deltadelta(838.025/rhoc),
                          -0.147637878)
        self.assert_close(iapws1995._phio_deltadelta(358/rhoc), -0.808994726)

    def test_phio_tau(self):
        """Tests _phio_tau()."""
        self.assert_close(iapws1995._phio_tau(Tc/500), 0.904611106e1)
        self.assert_close(iapws1995._phio_tau(Tc/647), 0.980343918e1)

    def test_phio_tautau(self):
        """Tests _phio_tautau()."""
        self.assert_close(iapws1995._phio_tautau(Tc/500), -0.193249185e1)
        self.assert_close(iapws1995._phio_tautau(Tc/647), -0.343316334e1)

    def test_phir(self):
        """Tests _phir()."""
        self.assert_close(iapws1995._phir(838.025/rhoc, Tc/500), -0.342693206e1)
        self.assert_close(iapws1995._phir(358/rhoc, Tc/647), -0.121202657e1)

    def test_phir_delta(self):
        """Tests _phir_delta."""
        self.assert_close(iapws1995._phir_delta(838.025/rhoc, Tc/500),
                          -0.364366650)
        self.assert_close(iapws1995._phir_delta(358/rhoc, Tc/647), -0.714012024)

    def test_phir_deltadelta(self):
        """Tests_phir_deltadelta()."""
        self.assert_close(iapws1995._phir_deltadelta(838.025/rhoc, Tc/500),
                          0.856063701)
        self.assert_close(iapws1995._phir_deltadelta(358/rhoc, Tc/647),
                          0.475730696)

    def test_phir_tau(self):
        """Tests _phir_tau."""
        self.assert_close(iapws1995._phir_tau(838.025/rhoc, Tc/500),
                          -0.581403435e1)
        self.assert_close(iapws1995._phir_tau(358/rhoc, Tc/647), -0.321722501e1)

    def test_phir_tautau(self):
        """Tests _phir_tautau()."""
        self.assert_close(iapws1995._phir_tautau(838.025/rhoc, Tc/500),
                          -0.223440737e1)
        self.assert_close(iapws1995._phir_tautau(358/rhoc, Tc/647),
                          -0.996029507e1)

    def test_phir_deltatau(self):
        """Tests _phir_deltatau."""
        self.assert_close(iapws1995._phir_deltatau(838.025/rhoc, Tc/500),
                          -0.112176915e1)
        self.assert_close(iapws1995._phir_deltatau(358/rhoc, Tc/647),
                          -0.133214720e1)


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
