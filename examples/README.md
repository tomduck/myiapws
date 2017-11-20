
Examples
========

Temperature-dependent curves:

  * cp.py - Isobaric heat capacity for liquid water at normal
    pressure.

  * alpha.py - Isobaric thermal expansion coefficient for 
    liquid water at normal pressure.

  * v.py - Specific volume for liquid water and ice at 
    normal pressure.

  * hsat.py - Saturation enthalpies for liquid water, ice
    and vapour.

  * l.py - Enthalpies of transformation for vaporization,
    sublimation and melting.

Phase diagrams:

  * pv.py - Pressure-volume isotherms for liquid water, vapour and
    ice.

  * Ts.py - Temperature-entropy isobars for liquid water and vapour.

  * mollier.py - Mollier diagram; isobars and isotherms on h-s axes.

  * coexistence-curves.py - Coexistence curves for vapor, liquid
    water and ice.


Discussion
----------

The IAPWS 1995 formulation provides functions X≡X(rho, T).  This presents a challenge, because we often do not know one or both of these variables a priori.

For example, suppose we want to plot Cp(T) at 101.325 kPa.  We can define an array of temperatures associated with the bottom axis.  However, to calculate Cp(rho, T) we need to also know the densities.  This is achieved by solving 

    p(rho, T) - 101.325 kPa = 0

for rho using Newton's method.  An estimate for the density is required, and this can often be obtained using the ideal gas law or saturation densities.
