# util.py: Utility functions for myiapws

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

"""Utility functions for myiapws."""

import functools

import numpy

# pylint: disable=invalid-name

# Sum function that allows functions to operate on arrays
# pylint: disable=redefined-builtin
sum = functools.partial(numpy.sum, axis=-1, keepdims=True)

def isscalar(x):
    """Determines if value is a scalar."""
    return numpy.isscalar(x) or numpy.asarray(x).shape in [(), (1,)]

def asscalar(x):
    """Returns value as a scalar."""
    return numpy.asscalar(numpy.asarray(x))

def arrayfunc(func):
    """Decorator to adapt functions for array input."""

    @functools.wraps(func)
    def wrapper(*args):
        """Wrapper for multi-variable input, single variable output."""
        args = [x if numpy.isscalar(x) else numpy.expand_dims(x, -1) \
                for x in args]
        ret = func(*args)
        return asscalar(ret) if numpy.isscalar(ret) else ret.squeeze()
    wrapper.__name__ = func.__name__

    return wrapper
