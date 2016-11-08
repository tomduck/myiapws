# iapws95.py: Library for IAPWS 1995 thermodynamic properties of water

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

"""IAPWS 1995 thermodynamic properties of ordinary water substance.

Overview:

  This package implements the IAWPS 1995 formulas for thermodynamic
  properties of liquid water and vapour.  Functions are of the form
  f(rho,T) where rho is the density (kg/m^3) and T is the temperature (K).

  Also included is a function rho(T) to calculate the saturation densities
  of liquid water and vapour.  The equations are given by IAPWS 1995, but
  the solution is owing to Akasaka (2008).

References:

  [1] Revised release on the IAPWS formulation 1995 for the thermodynamic
      properties of ordinary water substance for general and scientific use

      IAPWS, 2014.

      http://iapws.org/relguide/IAPWS95-2014.pdf

  [2] The IAPWS formulation 1995 for the thermodynamic properties of
      ordinary water substance for general and scientific use

      Wagner, W., and A. Pruss, J. Phys. Chem. Ref. Data, 31, 387-535, 2002.

      http://www.nist.gov/data/PDFfiles/jpcrd617.pdf

  [3] A reliable and useful method to determine the saturation state from
      Helmholtz energy equations of state

      Akasaka, R., J. Thermal. Sci. Tech., 3, 442-451, 2008.

      https://www.jstage.jst.go.jp/article/jtst/3/3/3_3_442/_pdf
"""

# pylint: disable=invalid-name

import functools

import numpy
from numpy import exp as _exp

from . import iapws92

_NMAX = 10          # The max number of iterations in the saturation calculation
_THRES = 1.e-12     # The convergence threshold in the saturation calculation
_DELTAMIN = 1.e-20  # The delta used in virial coefficient calculations


# HELPER FUNCTIONS -----------------------------------------------------------

def _asscalar(x):
    """Returns length-1 array value and passes through scalars."""
    return numpy.asscalar(numpy.asarray(x))


# Sum function that allows functions to operate on arrays
_sum = functools.partial(numpy.sum, axis=-1, keepdims=True)


def _arrayfunc(n):
    """Decorator factory function to adapt functions for array input."""

    def decorator(func):
        """Decorator to adapt functions for array input."""

        # There are different wrapper choices depending upon the
        # function's trace

        @functools.wraps(func)
        def wrapper1(rho, T):
            """Wrapper for rho,T input, single field output."""
            rho = rho if numpy.isscalar(rho) else numpy.expand_dims(rho, -1)
            T = T if numpy.isscalar(T) else numpy.expand_dims(T, -1)
            ret = func(rho, T)
            return _asscalar(ret) if numpy.isscalar(ret) else ret.squeeze()
        wrapper1.__name__ = func.__name__

        @functools.wraps(func)
        def wrapper2(T):
            """Wrapper for T input, single field output."""
            T = T if numpy.isscalar(T) else numpy.expand_dims(T, -1)
            ret = func(T)
            return _asscalar(ret) if numpy.isscalar(ret) else ret.squeeze()
        wrapper2.__name__ = func.__name__

        return wrapper1 if n == 1 else wrapper2

    return decorator


def _critical_value(v):
    """Decorator factory function to handle critical-point special case."""

    def decorator(func):
        """Decorator to handle critical-point special case."""

        @functools.wraps(func)
        def wrapper(rho, T):
            """Special computation for critical point values."""

            # Create a mask identifying critical points
            mask = numpy.logical_and(numpy.asarray(rho) == rhoc,
                                     numpy.asarray(T) == Tc)
            if mask.any():
                if v == None:
                    raise NotImplementedError('Critical value unknown')

                # Replace the critical density with something that will compute
                rho = numpy.where(mask, rhoc*1.1, rho)

            ret = func(rho, T)

            # Use the mask to insert the known critical point values and return
            if mask.any():
                ret = numpy.where(mask, v, ret)
            return _asscalar(ret) if numpy.isscalar(ret) else ret

        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# PUBLIC API -----------------------------------------------------------------

# Triple-point values
Tt = 273.16        # Temperature (K)
pt = 611.654771    # Pressure (Pa); Ref. [1] footnote 1 (pg. 4)

# Critical-point values
Tc = 647.096       # Temperature (K);  Ref. [1] eq. 1
rhoc = 322.        # Density (kg/m^3);  Ref. [1] eq. 2
pc = 22.064e6      # Pressure (Pa); Ref. [2] pg. 494
sc = 4.407e3       # Entropy (kJ/kg/K); Ref. [2] pg. 494
hc = 2084.26e3     # Enthalpy (kJ/kg); Ref. [2] pg. 494
uc = hc - pc/rhoc  # Internal energy (kJ/kg)
fc = uc-Tc*sc      # Helmholtz potential (kJ/kg)
cvc = numpy.inf    # Isochoric heat capacity; see Ref. [2] pg. 424 sec. 5.4.4
cpc = numpy.inf    # Isobaric heat capacity; see Ref. [2] pg. 424 sec. 5.4.4
wc = 0.            # Speed of sound; see Ref. [2] pg. 424 sec. 5.4.4
muc = None         # Joule-Thomson coefficient; unknown
deltaTc = None     # Isothermal throttling coefficient; unknown
betasc = None      # Isentropic temperature-pressure coefficient; unknown

R = 461.51805      # Specific gas constant (J/kg/K);  Ref. [1] eq. 3


## IAPWS95 Functions (ref. [1]) ##

# Table 3

@_arrayfunc(1)
@_critical_value(fc)
def f(rho, T):
    """Helmholtz potential (J/kg)"""
    delta, tau = rho/rhoc, Tc/T
    return R*T * _phi(delta, tau)

@_arrayfunc(1)
@_critical_value(pc)
def p(rho, T):
    """Pressure (Pa)"""
    delta, tau = rho/rhoc, Tc/T
    return rho*R*T * (1+delta*_phir_delta(delta, tau))

@_arrayfunc(1)
@_critical_value(uc)
def u(rho, T):
    """Specific internal energy (J/kg)"""
    delta, tau = rho/rhoc, Tc/T
    return R*T * tau * (_phio_tau(tau) + _phir_tau(delta, tau))

@_arrayfunc(1)
@_critical_value(sc)
def s(rho, T):
    """Specific entropy (J/kg/K)"""
    delta, tau = rho/rhoc, Tc/T
    return R * (tau * (_phio_tau(tau) + _phir_tau(delta, tau)) - \
            _phio(delta, tau) - _phir(delta, tau))

@_arrayfunc(1)
@_critical_value(hc)
def h(rho, T):
    """Specific enthalpy (J/kg)"""
    delta, tau = rho/rhoc, Tc/T
    return R*T * (1 + tau*(_phio_tau(tau) + _phir_tau(delta, tau)) +
                  delta*_phir_delta(delta, tau))

@_arrayfunc(1)
@_critical_value(cvc)
def cv(rho, T):
    """Isochoric specific heat capacity (J/kg/K)"""
    delta, tau = rho/rhoc, Tc/T
    return -R * tau**2 * (_phio_tautau(tau) + _phir_tautau(delta, tau))

@_arrayfunc(1)
@_critical_value(cpc)
def cp(rho, T):
    """Isobaric specific heat capacity (J/kg/K)"""
    delta, tau = rho/rhoc, Tc/T
    return -R * tau**2 * (_phio_tautau(tau) + _phir_tautau(delta, tau)) + \
      R*(1+delta*_phir_delta(delta, tau)-\
         delta*tau*_phir_deltatau(delta, tau))**2 / \
      (1+2*delta*_phir_delta(delta, tau)+delta**2*_phir_deltadelta(delta, tau))

@_arrayfunc(1)
@_critical_value(wc)
def w(rho, T):
    """Speed of sound (m/s)"""
    delta, tau = rho/rhoc, Tc/T
    return numpy.sqrt(R*T * (1 + 2*delta*_phir_delta(delta, tau)+\
                        delta**2*_phir_deltadelta(delta, tau)-\
                        (1+delta*_phir_delta(delta, tau)-\
                         delta*tau*_phir_deltatau(delta, tau))**2/\
                         (tau**2*(_phio_tautau(tau)+_phir_tautau(delta, tau)))))

@_arrayfunc(1)
@_critical_value(muc)
def mu(rho, T):
    """Joule-Thomson coefficient (K/Pa)"""
    delta, tau = rho/rhoc, Tc/T
    return -(delta*_phir_delta(delta, tau)+\
             delta**2*_phir_deltadelta(delta, tau)+\
             delta*tau*_phir_deltatau(delta, tau))/\
             ((1+delta*_phir_delta(delta, tau)-delta*tau*\
               _phir_deltatau(delta, tau))**2 - \
              tau**2*(_phio_tautau(tau)+_phir_tautau(delta, tau)) * \
              (1 + 2*delta*_phir_delta(delta, tau) + \
               delta**2*_phir_deltadelta(delta, tau)))/(R*rho)

@_arrayfunc(1)
@_critical_value(deltaTc)
def deltaT(rho, T):
    """Specific isothermal throttling coefficient (J/Pa)"""
    delta, tau = rho/rhoc, Tc/T
    return (1 - (1+delta*_phir_delta(delta, tau)-\
                 delta*tau*_phir_deltatau(delta, tau))/\
                 (1+2*delta*_phir_delta(delta, tau)+\
                  delta**2*_phir_deltadelta(delta, tau)))/rho

@_arrayfunc(1)
@_critical_value(betasc)
def betas(rho, T):
    """Isentropic temperature-pressure coefficient (K/Pa)"""
    delta, tau = rho/rhoc, Tc/T
    return (1 + delta*_phir_delta(delta, tau)-\
            delta*tau*_phir_deltatau(delta, tau))/\
            ((1+delta*_phir_delta(delta, tau)-\
              delta*tau*_phir_deltatau(delta, tau))**2-\
              tau**2*(_phio_tautau(tau)+_phir_tautau(delta, tau))*\
              (1+2*delta*_phir_delta(delta, tau)+\
                  delta**2*_phir_deltadelta(delta, tau)))/(rho*R)

@_arrayfunc(2)
def B(T):
    """Second virial coefficient (m^3/kg)"""
    return _phir_delta(_DELTAMIN, Tc/T)/rhoc

@_arrayfunc(2)
def C(T):
    """Third virial coefficient (m^6/kg^2)"""
    return _phir_deltadelta(_DELTAMIN, Tc/T)/rhoc**2


## Saturation functions (ref. [3]) ##

def rhosat(T):
    """Saturation liquid and vapour densities (kg/m^3)"""

    # This function does its own array and critical point management (rather
    # than by using decorators) because of its unique trace

    T = T if numpy.isscalar(T) else numpy.expand_dims(T, -1)

    # Create a mask for the critical temperature
    if numpy.any(numpy.logical_and(T < Tt, T > Tc)):
        raise ValueError('T must be between Tt and Tc')

    # Replace the critical value with something that computes
    mask = T == Tc
    T = numpy.where(mask, Tt, T)

    # Initial guesses for delta' and delta''.  In practice the guesses have
    # to be accurate near the critical point.
    dp = iapws92.rhop(T)/rhoc
    dpp = iapws92.rhopp(T)/rhoc

    tau = Tc/T

    flag = False  # Success flag

    for i in range(_NMAX):  # pylint: disable=unused-variable

        Delta_K = _K(dpp, tau) - _K(dp, tau)
        Delta_J = _J(dpp, tau) - _J(dp, tau)

        # Test for convergence
        if numpy.all(numpy.fabs(Delta_K) + numpy.fabs(Delta_J) < _THRES):
            flag = True
            break

        # Eq. 25
        Delta = _J_delta(dpp, tau)*_K_delta(dp, tau) - \
                _J_delta(dp, tau)*_K_delta(dpp, tau)

        # Eqs. 19 and 20
        dp += 1./Delta*(Delta_K*_J_delta(dpp, tau)-Delta_J*_K_delta(dpp, tau))
        dpp += 1./Delta*(Delta_K*_J_delta(dp, tau)-Delta_J*_K_delta(dp, tau))

    if flag:
        rhol = numpy.where(mask, rhoc, dp*rhoc)
        rhov = numpy.where(mask, rhoc, dpp*rhoc)
        return _asscalar(rhol) if numpy.isscalar(rhol) else rhol.squeeze(), \
          _asscalar(rhov) if numpy.isscalar(rhov) else rhov.squeeze(), \

    else:
        raise RuntimeError('Saturation calculation not converging')


## Identities ##

# These thermodynamic functions are obtained from cp, cv and w (above) by
# using thermodynamic identities.

def kappa_T(rho, T):
    """Isothermal compressibility (/Pa)"""
    return cp(rho, T)/cv(rho, T)/(rho*w(rho, T)**2)

def kappa_s(rho, T):
    """Isentropic compressibility (/Pa)"""
    return 1/(rho*w(rho, T)**2)

def alpha_V(rho, T):
    """Volumetric thermal expansion coefficient (/K)"""
    # https://en.wikipedia.org/wiki/Joule%E2%80%93Thomson_effect
    return (mu(rho, T)*rho*cp(rho, T)+1)/T


# PRIVATE --------------------------------------------------------------------

## IAPWS95 coefficients (ref. [1]) ##

# Table 1

_no = numpy.array([-8.3204464837497, 6.6832105275932, 3.00632, 0.012436,
                   0.97315, 1.27950, 0.96956, 0.24873])

_gammao = numpy.array([numpy.nan, numpy.nan, numpy.nan,
                       1.28728967, 3.53734222, 7.74073708,
                       9.24437796, 27.5075105])


# Table 2 coefficients

_c = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
                  numpy.nan, numpy.nan,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                  3, 3, 3, 3, 4, 6, 6, 6, 6])

_d = numpy.array([1, 1, 1, 2, 2, 3, 4, 1, 1, 1, 2, 2, 3, 4, 4, 5, 7, 9, 10,
                  11, 13, 15, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 9, 9, 9,
                  9, 9, 10, 10, 12, 3, 4, 4, 5, 14, 3, 6, 6, 6, 3, 3, 3])

_t = numpy.array([-0.5, 0.875, 1, 0.5, 0.75, 0.375, 1, 4, 6, 12, 1, 5, 4, 2,
                  13, 9, 3, 4, 11, 4, 13, 1, 7, 1, 9, 10, 10, 3, 7, 10, 10, 6,
                  10, 10, 1, 2, 3, 4, 8, 6, 9, 8, 16, 22, 23, 23, 10, 50, 44,
                  46, 50, 0, 1, 4])

# pylint: disable=bad-whitespace,bad-continuation
_n = numpy.array([ 0.12533547935523e-1,   0.78957634722828e+1,
                  -0.87803203303561e+1,   0.31802509345418,
                  -0.26145533859358,     -0.78199751687981e-2,
                   0.88089493102134e-2,  -0.66856572307965,
                   0.20433810950965,     -0.66212605039687e-4,
                  -0.19232721156002,     -0.25709043003438,
                   0.16074868486251,     -0.40092828925807e-1,
                   0.39343422603254e-6,  -0.75941377088144e-5,
                   0.56250979351888e-3,  -0.15608652257135e-4,
                   0.11537996422951e-8,   0.36582165144204e-6,
                  -0.13251180074668e-11, -0.62639586912454e-9,
                  -0.10793600908932,      0.17611491008752e-1,
                   0.22132295167546,     -0.40247669763528,
                   0.58083399985759,      0.49969146990806e-2,
                  -0.31358700712549e-1,  -0.74315929710341,
                   0.47807329915480,      0.20527940895948e-1,
                  -0.13636435110343,      0.14180634400617e-1,
                   0.83326504880713e-2,  -0.29052336009585e-1,
                   0.38615085574206e-1,  -0.20393486513704e-1,
                  -0.16554050063734e-2,   0.19955571979541e-2,
                   0.15870308324157e-3,  -0.16388568342530e-4,
                   0.43613615723811e-1,   0.34994005463765e-1,
                  -0.76788197844621e-1,   0.22446277332006e-1,
                  -0.62689710414685e-4,  -0.55711118565645e-9,
                  -0.19905718354408,      0.31777497330738,
                  -0.11841182425981,     -0.31306260323435e+2,
                   0.31546140237781e+2,  -0.25213154341695e+4,
                  -0.14874640856724,      0.31806110878444])


_a = numpy.array([3.5, 3.5])

_b = numpy.array([0.85, 0.95])

_B = numpy.array([0.2, 0.2])


_alpha = numpy.array([20., 20, 20])

_beta1 = numpy.array([150., 150, 250])

_gamma = numpy.array([1.21, 1.21, 1.25])

_epsilon = numpy.array([1.]*3)


_C = numpy.array([28., 32])

_D = numpy.array([700., 800])

_A = numpy.array([0.32, 0.32])

_beta2 = numpy.array([0.3, 0.3])


## IAPWS95 functions (ref [1]) ##

def _phi(delta, tau):  # Eq. 4
    """Dimensionless Helmholtz potential."""
    return _phio(delta, tau) + _phir(delta, tau)


# Table 4: The ideal-gas part phio of the dimensionless Helmholtz potential
#          and its derivatives

def _phio(delta, tau):
    """Ideal gas part of the dimensionless Helmholtz potential."""
    return numpy.log(delta) + _no[0] + _no[1]*tau + _no[2]*numpy.log(tau) + \
      _sum(_no[3:8]*numpy.log(1-_exp(-_gammao[3:8]*tau)))

def _phio_delta(delta):
    """Partial derivative of phi_o wrt delta."""
    return 1./delta

def _phio_deltadelta(delta):
    """Second partial derivative of phi_o wrt delta."""
    return -1./delta**2

def _phio_tau(tau):
    """Partial derivative of phi_o wrt tau."""
    return _no[1] + _no[2]/tau + \
      _sum(_no[3:8]*_gammao[3:8]*((1-_exp(-_gammao[3:8]*tau))**(-1) - 1))

def _phio_tautau(tau):
    """Second partial derivative of phi_o wrt delta."""
    return -_no[2]*tau**-2 - \
      _sum(_no[3:8]*_gammao[3:8]**2*_exp(-_gammao[3:8]*tau)*\
                (1-_exp(-_gammao[3:8]*tau))**(-2))

def _phio_deltatau():
    """Mixed second partial derivative of phi_o wrt delta and tau."""
    return 0


# Table 5: The residual part phir of the dimensionless Helmholtz potential
#          and its derivatives

def _phir(delta, tau):
    """Residual part of the dimensionless Helmholtz potential."""
    psi = _exp(-_C*(delta-1)**2 - _D*(tau-1)**2)
    theta = (1-tau) + _A*((delta-1)**2)**(1/(2*_beta2))
    Delta = theta**2 + _B*((delta-1)**2)**_a
    return _sum(_n[0:7]*delta**_d[0:7]*tau**_t[0:7]) \
      + _sum(_n[7:51]*delta**_d[7:51]*tau**_t[7:51]*\
                  _exp(-delta**_c[7:51])) \
      + _sum(_n[51:54]*delta**_d[51:54]*tau**_t[51:54] * \
                  _exp(-_alpha*(delta-_epsilon)**2 - _beta1*(tau-_gamma)**2)) \
      + _sum(_n[54:56]*Delta**_b*delta*psi)

def _phir_delta(delta, tau):
    """Partial derivative of phi_r wrt delta."""
    psi = _exp(-_C*(delta-1)**2 - _D*(tau-1)**2)
    theta = (1-tau) + _A*((delta-1)**2)**(1/(2*_beta2))
    Delta = theta**2 + _B*((delta-1)**2)**_a
    return _sum(_n[0:7]*_d[0:7]*delta**(_d[0:7]-1)*tau**_t[0:7]) + \
      _sum(_n[7:51]*_exp(-delta**_c[7:51]) * \
                delta**(_d[7:51]-1)*tau**_t[7:51] * \
                (_d[7:51]-_c[7:51]*delta**_c[7:51])) + \
      _sum(_n[51:54]*delta**_d[51:54]*tau**_t[51:54] * \
                _exp(-_alpha*(delta-_epsilon)**2 - \
                    _beta1*(tau-_gamma)**2) *
                    (_d[51:54]/delta - 2*_alpha*(delta-_epsilon))) +\
      _sum(_n[54:56]*(Delta**_b*(psi+delta*_pder_psi_delta(psi, delta)) +
                          _pder_Deltab_delta(Delta, delta, theta)*delta*psi))

def _phir_deltadelta(delta, tau):
    """Second partial derivative of phi_r wrt delta."""
    psi = _exp(-_C*(delta-1)**2 - _D*(tau-1)**2)
    theta = (1-tau) + _A*((delta-1)**2)**(1/(2*_beta2))
    Delta = theta**2 + _B*((delta-1)**2)**_a
    return _sum(_n[0:7]*_d[0:7]*(_d[0:7]-1)*delta**(_d[0:7]-2)
                     *tau**_t[0:7])+ \
      _sum(_n[7:51]*_exp(-delta**_c[7:51])*\
                (delta**(_d[7:51]-2)*tau**_t[7:51]*\
                 ((_d[7:51]-_c[7:51]*delta**_c[7:51])*\
                  (_d[7:51]-1-_c[7:51]*delta**_c[7:51])-_c[7:51]**2*\
                 delta**_c[7:51]))) +\
      _sum(_n[51:54]*tau**_t[51:54]*\
                _exp(-_alpha*(delta-_epsilon)**2-_beta1*(tau-_gamma)**2)*\
                (-2*_alpha*delta**_d[51:54]+4*_alpha**2*delta**_d[51:54]*\
                 (delta-_epsilon)**2-4*_d[51:54]*_alpha*delta**(_d[51:54]-1)*\
                (delta-_epsilon)+\
                _d[51:54]*(_d[51:54]-1)*delta**(_d[51:54]-2))) + \
      _sum(_n[54:56]*(Delta**_b*(2*_pder_psi_delta(psi, delta)+\
          delta*_pder2_psi_delta2(psi, delta))+\
          2*_pder_Deltab_delta(Delta, delta, theta)*\
          (psi+delta*_pder_psi_delta(psi, delta))+\
          _pder2_Deltab_delta2(Delta, delta, theta)*delta*psi))

def _phir_tau(delta, tau):
    """Partial derivative of phi_r wrt tau."""
    psi = _exp(-_C*(delta-1)**2 - _D*(tau-1)**2)
    theta = (1-tau) + _A*((delta-1)**2)**(1/(2*_beta2))
    Delta = theta**2 + _B*((delta-1)**2)**_a
    return _sum(_n[0:7]*_t[0:7]*delta**_d[0:7]*tau**(_t[0:7]-1)) +\
      _sum(_n[7:51]*_t[7:51]*delta**_d[7:51]*tau**(_t[7:51]-1) * \
                _exp(-delta**_c[7:51])) + \
      _sum(_n[51:54]*delta**_d[51:54]*tau**_t[51:54] * \
                _exp(-_alpha*(delta-_epsilon)**2-_beta1*(tau-_gamma)**2) * \
                (_t[51:54]/tau-2*_beta1*(tau-_gamma))) + \
      _sum(_n[54:56]*delta*(_pder_Deltab_tau(Delta, theta)*psi + \
                                Delta**_b*_pder_psi_tau(tau, psi)))

def _phir_tautau(delta, tau):
    """Second partial derivative of phi_r wrt delta."""
    psi = _exp(-_C*(delta-1)**2 - _D*(tau-1)**2)
    theta = (1-tau) + _A*((delta-1)**2)**(1/(2*_beta2))
    Delta = theta**2 + _B*((delta-1)**2)**_a
    return _sum(_n[0:7]*_t[0:7]*(_t[0:7]-1)*delta**_d[0:7]*\
                     tau**(_t[0:7]-2)) + \
      _sum(_n[7:51]*_t[7:51]*(_t[7:51]-1)*delta**_d[7:51] * \
                tau**(_t[7:51]-2)*_exp(-delta**_c[7:51])) + \
      _sum(_n[51:54]*delta**_d[51:54]*tau**_t[51:54] * \
                _exp(-_alpha*(delta-_epsilon)**2-_beta1*(tau-_gamma)**2) * \
            ((_t[51:54]/tau-2*_beta1*(tau-_gamma))**2-_t[51:54]/tau**2-2*
             _beta1)) + \
      _sum(_n[54:56]*delta * (_pder2_Deltab_tau2(Delta, theta)*psi + \
                2*_pder_Deltab_tau(Delta, theta)*_pder_psi_tau(tau, psi) + \
                Delta**_b*_pder2_psi_tau2(tau, psi)))

def _phir_deltatau(delta, tau):
    """Mixed second partial derivative of phi_r wrt delta and tau."""
    psi = _exp(-_C*(delta-1)**2 - _D*(tau-1)**2)
    theta = (1-tau) + _A*((delta-1)**2)**(1/(2*_beta2))
    Delta = theta**2 + _B*((delta-1)**2)**_a
    return _sum(_n[0:7]*_d[0:7]*_t[0:7]*delta**(_d[0:7]-1)*\
                     tau**(_t[0:7]-1)) + \
      _sum(_n[7:51]*_t[7:51]*delta**(_d[7:51]-1)*tau**(_t[7:51]-1)* \
                (_d[7:51]-_c[7:51]*delta**_c[7:51])*_exp(-delta**_c[7:51])) +\
      _sum(_n[51:54]*delta**_d[51:54]*tau**_t[51:54]* \
                _exp(-_alpha*(delta-_epsilon)**2-_beta1*(tau-_gamma)**2)* \
                (_d[51:54]/delta-2*_alpha*(delta-_epsilon))* \
                (_t[51:54]/tau-2*_beta1*(tau-_gamma))) + \
      _sum(_n[54:56]*(Delta**_b*(_pder_psi_tau(tau, psi)+\
          delta*_pder2_psi_deltatau(delta, tau, psi))+\
          delta*_pder_Deltab_delta(Delta, delta, theta)*\
          _pder_psi_tau(tau, psi)+_pder_Deltab_tau(Delta, theta)*\
          (psi+delta*_pder_psi_delta(psi, delta))+\
          _pder2_Deltab_deltatau(Delta, delta, theta)*delta*psi))


# Derivatives of the distance function Delta**b

def _pder_Deltab_delta(Delta, delta, theta):
    """Partial derivative of distance function wrt delta."""
    return _b*Delta**(_b-1) * _pder_Delta_delta(delta, theta)

def _pder2_Deltab_delta2(Delta, delta, theta):
    """Second partial derivative of distance function wrt delta."""
    return _b*(Delta**(_b-1)*_pder2_Delta_delta2(delta, theta)+\
              (_b-1)*Delta**(_b-2)*_pder_Delta_delta(delta, theta)**2)

def _pder_Deltab_tau(Delta, theta):
    """Partial derivative of distance function wrt tau."""
    return -2*theta*_b*Delta**(_b-1)

def _pder2_Deltab_tau2(Delta, theta):
    """Second partial derivative of distance function wrt tau."""
    return 2*_b*Delta**(_b-1) + 4*theta**2*_b*(_b-1)*Delta**(_b-2)

def _pder2_Deltab_deltatau(Delta, delta, theta):
    """Mixed second partial derivative of distance function
    wrt delta and tau."""
    return -_A*_b*2/_beta2*Delta**(_b-1)*(delta-1)* \
      ((delta-1)**2)**(1/(2*_beta2)-1)-2*theta*_b*(_b-1)*Delta**(_b-2)* \
      _pder_Delta_delta(delta, theta)


# ... with

def _pder_Delta_delta(delta, theta):
    """Partial derivative of Delta wrt delta."""
    return (delta-1)*(_A*theta*2/_beta2*((delta-1)**2)**(1/(2*_beta2)-1) + \
                      2*_B*_a*((delta-1)**2)**(_a-1))

def _pder2_Delta_delta2(delta, theta):
    """Second partial derivative of Delta wrt delta."""
    return 1/(delta-1)*_pder_Delta_delta(delta, theta)+(delta-1)**2*\
      (4*_B*_a*(_a-1)*((delta-1)**2)**(_a-2)+2*_A**2*(1/_beta2)**2*\
       (((delta-1)**2)**(1/(2*_beta2)-1))**2 + _A*theta*4./_beta2*\
      (1/(2*_beta2)-1)*((delta-1)**2)**(1/(2*_beta2)-2))


# Derivatives of the exponential function psi

def _pder_psi_delta(psi, delta):
    """Partial derivative of psi wrt delta."""
    return -2*_C * (delta-1) * psi

def _pder2_psi_delta2(psi, delta):
    """Second partial derivative of psi wrt delta."""
    return (2*_C*(delta-1)**2-1)*2*_C*psi

def _pder_psi_tau(tau, psi):
    """Partial derivative of psi wrt tau."""
    return -2*_D*(tau-1)*psi

def _pder2_psi_tau2(tau, psi):
    """Partial derivative of psi wrt tau."""
    return (2*_D*(tau-1)**2-1)*2*_D*psi

def _pder2_psi_deltatau(delta, tau, psi):
    """Mixed second partial derivative of psi wrt delta and tau."""
    return 4*_C*_D*(delta-1)*(tau-1)*psi


## Saturation functions (ref. [3]) ##

def _J(delta, tau):  # Eq. 21
    """Newton-Raphson J coefficient."""
    return delta*(1+delta*_phir_delta(delta, tau))

def _K(delta, tau):  # Eq. 22
    """Newton-Raphson K coefficient."""
    return delta*_phir_delta(delta, tau) + _phir(delta, tau) + numpy.log(delta)

def _J_delta(delta, tau):  # Eq. 23
    """Partial derivative of Newton-Raphson J coefficient wrt tau."""
    return 1 + 2*delta*_phir_delta(delta, tau) + \
      delta**2*_phir_deltadelta(delta, tau)

def _K_delta(delta, tau):  # Eq. 24
    """Partial derivative of Newton-Raphson K coefficient wrt tau."""
    return 2*_phir_delta(delta, tau) + delta*_phir_deltadelta(delta, tau) + \
      1/delta
