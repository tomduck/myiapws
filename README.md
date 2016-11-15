
myiapws
=======

Myiapws is a python library for the calculation of thermophysical properties of water.  It is an implementation of certain formulations from the [International Association for the Properties of Water and Steam](http://www.iapws.org/) (IAPWS).

The primary use of the library is as an educational tool.  The publications describing the formulations map directly to the code.  The interface is simple, fully pythonic and numpy-capable.

Developers interested in an advanced thermodynamics package covering over 100 species should check out [COOLPROP](http://www.coolprop.org/).


Installation
------------

Install into python3 (as root) by executing

~~~
# pip3 install git+https://github.com/tomduck/myiapws.git
~~~

The library should be tested (as a normal user) by executing

~~~
$ cd test
$ ./test.py
~~~

The unit tests are based on "computer program verification" values provided by IAPWS.


Modules
-------

The available modules are as follows:

  * `myiapws.iapws1992`: Thermodynamic properties on the coexistence curve of liquid water and vapor.  ([REF](http://www.iapws.org/relguide/supsat.pdf))

  * `myiapws.iapws1995`: Thermodynamic properties of ordinary water substance (liquid and gas).  The fundamental equation is in Helmholtz representation, and so all properties are functions of temperature and density. ([REF](http://iapws.org/relguide/IAPWS95-2014.pdf))

  * `myiapws.iapws2006`: Thermodynamic properties of ice Ih.  The fundamental equation is in Gibbs representation, and so all properties are functions of temperature and pressure. ([REF](http://iapws.org/relguide/Ice-Rev2009.pdf))

  * `myiapws.iapws2011`: Melting and sublimation pressures of liquid water and ice as a function of temperature. ([REF](http://www.iapws.org/relguide/MeltSub2011.pdf))

The modules are internally documented.  Execute `help(<module-name>)` at the python prompt to get full descriptions of each API.


Examples
--------

It is not trivial to draw thermodynamic state diagrams from the fundamental equations and their derivatives.
