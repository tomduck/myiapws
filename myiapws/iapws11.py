#! /usr/bin/env python3

# iapws11.py: Library for IAPWS 2011 melting and sublimation curves of water
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

"""IAPWS 2011 melting and sublimation curves for water.

Overview:

  This package implements the IAWPS 2011 formulas for the melting and
  sublimation curves of water.

References:

  [1] Revised release on the pressure along the melting and sublimation
      curves of ordinary water substance.

      IAPWS, 2011.

      http://www.iapws.org/relguide/MeltSub2011.pdf
"""

def pmelt_iceIh(T):
    """Ice Ih melting pressure (MPa) for temperature (K)"""

    T = numpy.array(T)
    if not numpy.logical_and(251.165<=T,T<=273.16).all():
        raise ValueError('251.165 <= T <= 273.16 K for ice Ih melting line')
    
    a = [0.119539337e7, 0.808183159e5, 0.333826860e4]
    b = [0.300000e1, 0.257500e2, 0.103750e3]

    Tstar = 273.16   # K
    pstar = 611.657  # Pa

    theta = T/Tstar
    
    return (1. + sum(a[i]*(1.-theta**b[i]) for i in range(3))) * pstar / 1.e6

    
def pmelt_iceIII(T):
    """Ice III melting pressure (MPa) for temperature (K)"""

    T = numpy.array(T)
    if not numpy.logical_and(251.165<=T,T<=256.164).all():
        raise ValueError('251.165 <= T <= 256.164 K for ice III melting line')

    Tstar = 251.165  # K
    pstar = 208.566  # MPa

    theta = T/Tstar

    return (1. - 0.299948*(1.-theta**60)) * pstar


def pmelt_iceV(T):
    """Ice V melting pressure (MPa) for temperature (K)"""

    T = numpy.array(T)
    if not numpy.logical_and(256.164<=T,T<=273.31).all():
        raise ValueError('256.164 <= T <= 273.31 for ice V melting line')

    Tstar = 256.164  # K
    pstar = 350.1    # MPa

    theta = T/Tstar

    return (1. - 1.18721*(1.-theta**8)) * pstar


def pmelt_iceVI(T):
    """Returns the ice VI melting pressure (MPa) for temperature (K)"""

    T = numpy.array(T)
    if not numpy.logical_and(273.31<=T,T<=355).all():
        raise ValueError('273.31 <= T <= 355 for ice VI melting line')

    Tstar = 273.31  # K
    pstar = 632.4  # MPa

    theta = T/Tstar

    return (1. - 1.07476*(1.-theta**4.6)) * pstar


def pmelt_iceVII(T):
    """Ice VII melting pressure (MPa) for temperature (K)"""

    T = numpy.array(T)
    if not numpy.logical_and(355<=T,T<=715).all():
        raise ValueError('355 <= T <= 715 for ice VII melting line')

    Tstar = 355.0   # K
    pstar = 2216.0  # MPa

    theta = T/Tstar

    return numpy.exp( (0.173683e1*(1.-1./theta)
                        - 0.544606e-1*(1.-theta**5)
                        + 0.806106e-7*(1.-theta**22)) ) * pstar

    
def psubl(T):
    """Ice sublimation pressure (MPa) for temperature (K)"""

    T = numpy.array(T)
    if not numpy.logical_and(50.<=T,T<=273.16).all():
        raise ValueError('50 <= T <= 273.16 K for ice sublimation line')

    a = [-0.212144006e2,0.273203819e2,-0.610598130e1]
    b = [0.333333333e-2,0.120666667e1,0.170333333e1]

    Tstar = 273.16   # K
    pstar = 611.657  # Pa

    theta = T/Tstar
    
    return numpy.exp(1./theta*sum(a[i]*theta**b[i] for i in range(3))) \
      * pstar / 1.e6
