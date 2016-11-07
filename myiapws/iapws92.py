#! /usr/bin/env python3

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

"""IAPWS 1992 saturation properties of fluid water.

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

import sys
import unittest
import functools

import numpy

_VERBOSE = False  # Flag for unit test verbosity


# Decorator to validate input
def _validate(f):
    @functools.wraps(f)
    def decorate(T):
        if numpy.logical_or(T<273.16,T>Tc).any():
            msg = '273.16 <= T <= 647.096 K for water vapour saturation line'
            raise ValueError(msg)
        ret=f(T)
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
    """Saturation vapour pressure (Pa)"""
    theta = T/Tc
    tau = 1.-theta
    return pc * numpy.exp( 1./theta *
        (_a[0]*tau+_a[1]*tau**1.5+_a[2]*tau**3+_a[3]*tau**3.5+_a[4]*tau**4
         +_a[5]*tau**7.5) )

@_validate
def dpdT(T):
    theta = T/Tc
    tau = 1.-theta
    return p(T)*\
      (-Tc/T**2 * (_a[0]*tau+_a[1]*tau**1.5+_a[2]*tau**3+_a[3]*tau**3.5+\
                   _a[4]*tau**4+_a[5]*tau**7.5) - \
       1./T * (_a[0]+_a[1]*1.5*tau**0.5+_a[2]*3*tau**2+_a[3]*3.5*tau**2.5+\
               _a[4]*4*tau**3+_a[5]*7.5*tau**6.5))

@_validate
def rhop(T):
    """Density of the saturated liquid (kg/m3)"""
    theta = T/Tc
    tau = 1.-theta
    return rhoc*(1+_b[0]*tau**(1./3)+_b[1]*tau**(2./3)+_b[2]*tau**(5./3)+\
                 _b[3]*tau**(16./3)+_b[4]*tau**(43./3)+_b[5]*tau**(110./3))
                 
@_validate
def rhopp(T):
    """Density of the saturated vapour (kg/m3)"""
    theta = T/Tc
    tau = 1.-theta
    return rhoc*numpy.exp(_c[0]*tau**(2./6)+_c[1]*tau**(4./6)+\
                          _c[2]*tau**(8./6)+_c[3]*tau**(18./6)+\
                          _c[4]*tau**(37./6)+_c[5]*tau**(71./6))

@_validate
def hp(T):
    """Specific enthalpy of the saturated liquid (J/kg)"""
    return _alpha(T) + T/rhop(T)*dpdT(T)

@_validate
def hpp(T):
    """Specific enthalpy of the saturated liquid (J/kg)"""
    return _alpha(T) + T/rhopp(T)*dpdT(T)

@_validate
def sp(T):
    """Specific entropy of the saturated liquid (J/kg/K)"""
    return _phi(T)+1/rhop(T)*dpdT(T)

@_validate
def spp(T):
    """Specific entropy of the saturated liquid (J/kg/K)"""
    return _phi(T)+1/rhopp(T)*dpdT(T)


# PRIVATE --------------------------------------------------------------------

# Coefficients
_a = numpy.array([-7.85951783,1.84408259,-11.7866497,22.6807411,-15.9618719,
                  1.80122502])
_b = numpy.array([1.99274064,1.09965342,-0.510839303,-1.75493479,-45.5170352,
                  -6.74694450e5])
_c = numpy.array([-2.03150240,-2.68302940,-5.38626492,-17.2991605,
                  -44.7586581,-63.9201063])
_d = numpy.array([-5.65134998e-8,2690.66631,127.287297,-135.003439,0.981825814])
_d_alpha = -1135.905627715
_d_phi = 2319.5246
_alpha0 = 1000.
_phi0 = _alpha0/Tc


# Functions

def _alpha(T):
    theta = T/Tc
    return _alpha0*(_d_alpha + _d[0]*theta**-19+_d[1]*theta+_d[2]*theta**4.5+\
                    _d[3]*theta**5+_d[4]*theta**54.5)

def _phi(T):
    theta = T/Tc
    return _phi0*(_d_phi+19./20*_d[0]*theta**-20+_d[1]*numpy.log(theta)+\
                  9./7*_d[2]*theta**3.5+5./4*_d[3]*theta**4+\
                  109./107*_d[4]*theta**53.5)


# Unit tests -----------------------------------------------------------------

class _Test_public(unittest.TestCase):

    T1 = 273.16
    T2 = 373.1243
    T3 = 647.096
    T = numpy.array([273.16,373.1243,647.096])
    
    def assert_close(self,v1,v2,rtol=1.e-8,atol=0.):
        isclose = numpy.isclose(v1,v2,rtol=rtol,atol=atol)
        if numpy.isscalar(isclose):
            self.assert_(isclose)
        else:
            self.assert_(isclose.all())

    def test_p(self):
        self.assert_close(p(self.T1),611.657,rtol=1.e-6)
        self.assert_close(p(self.T2),0.101325e6,rtol=1.e-6)
        self.assert_close(p(self.T3),22.064e6,rtol=1.e-5)
        self.assert_close(p(self.T),[611.657,0.101325e6,22.064e6],rtol=1.e-5)

        # Make sure masked values don't cause an error
        self.assertEqual(p(numpy.ma.masked_values(Tc,Tc)).mask,True)

    def test_dpdT(self):
        self.assert_close(dpdT(self.T1),44.436693,rtol=1.e-8)
        self.assert_close(dpdT(self.T2),3.616e3,rtol=2.e-4)
        self.assert_close(dpdT(self.T3),268.e3,rtol=1.e-3)
        self.assert_close(dpdT(self.T),[44.436693,3.616e3,268.e3],rtol=1.e-3)

    def test_rhop(self):
        self.assert_close(rhop(self.T1),999.789,rtol=1.e-6)
        self.assert_close(rhop(self.T2),958.365,rtol=1.e-6)
        self.assert_close(rhop(self.T3),322.,rtol=1.e-3)
        self.assert_close(rhop(self.T),[999.7897,958.365,322.],rtol=1.e-3)

    def test_rhopp(self):
        self.assert_close(rhopp(self.T1),0.00485426,rtol=1.e-6)
        self.assert_close(rhopp(self.T2),0.597586,rtol=1.e-6)
        self.assert_close(rhopp(self.T3),322.,rtol=1.e-3)
        self.assert_close(rhopp(self.T),[0.00485426,0.597586,322.],rtol=1.e-3)

    def test_hp(self):
        self.assert_close(hp(self.T1),0.611786,rtol=1.e-6)
        self.assert_close(hp(self.T2),419.05e3,rtol=2.e-5)
        self.assert_close(hp(self.T3),2086.6e3,rtol=2.e-5)
        self.assert_close(hp(self.T),[0.611786,419.05e3,2086.6e3],rtol=2.e-5)

    def test_hpp(self):
        self.assert_close(hpp(self.T1),2500.5e3,rtol=2.e-5)
        self.assert_close(hpp(self.T2),2675.7e3,rtol=1.e-5)
        self.assert_close(hpp(self.T3),2086.6e3,rtol=2.e-5)
        self.assert_close(hpp(self.T),[2500.5e3,2675.7e3,2086.6e3],rtol=2.e-5)

    def test_sp(self):
        self.assert_close(sp(self.T1),0.,atol=1.e-5)
        self.assert_close(sp(self.T2),1.307e3,rtol=1.e-4)
        self.assert_close(sp(self.T3),4.410e3,rtol=1.e-4)
        self.assert_close(sp(self.T),[0.,1.307e3,4.410e3],rtol=1.e-4,atol=1.e-5)
    
    def test_sp(self):
        self.assert_close(spp(self.T1),9.154e3,rtol=1.e-4)
        self.assert_close(spp(self.T2),7.355e3,rtol=1.e-4)
        self.assert_close(spp(self.T3),4.410e3,rtol=1.e-4)
        self.assert_close(spp(self.T),[9.154e3,7.355e3,4.410e3],rtol=1.e-4)

                    
class _Test_private(unittest.TestCase):

    T1 = 273.16
    T2 = 373.1243
    T3 = 647.096

    def assert_close(self,v1,v2,rtol=1.e-8,atol=0.):
        isclose = numpy.isclose(v1,v2,rtol=rtol,atol=atol)
        if numpy.isscalar(isclose):
            self.assert_(isclose)
        else:
            self.assert_(isclose.all())

    def test_alpha(self):
        self.assert_close(_alpha(self.T1),-11.529101)
        self.assert_close(_alpha(self.T2),417.65e3,rtol=1.e-5)
        self.assert_close(_alpha(self.T3),1548e3,rtol=1.e-4)

    def test_phi(self):
        self.assert_close(_phi(self.T1),-0.04,rtol=2.e-1)
        self.assert_close(_phi(self.T2),1.303e3,rtol=2.e-4)
        self.assert_close(_phi(self.T3),3.578e3,rtol=1.e-4)

                    
def _run_tests():
    Nerrors = 0
    Nfailures = 0

    for Test in [_Test_public,_Test_private]:
        suite = unittest.makeSuite(Test)
        result = unittest.TextTestRunner(verbosity=2 if _VERBOSE else 1)\
          .run(suite)
        Nerrors += len(result.errors)
        Nfailures += len(result.failures)

    if Nerrors or Nfailures:
        print('\n\nSummary: %d errors and %d failures reported\n'%\
            (Nerrors,Nfailures))

    return Nerrors+Nfailures


if __name__=='__main__':
    sys.exit(_run_tests())
