#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import numpy.ma as ma
from collections import namedtuple

from mra_sympy import MRA


TWO_PI = 2.*math.pi

INTEGRAL_FILE = "wavelet_data/wavelet_base%d_power%d_i%d_%s.dat"


class Point(namedtuple('Point', ['x', 'y', 'z'])):
    __slots__ = ()
    def __new__(cls, x, y, z):
        x, y, z = [int(round(i)) for i in (x, y, z)]
        return super().__new__(cls, x, y, z)
    def __add__(self, other):
        if not isinstance(other, Point):
            other = Point._make(other)
        return Point(self.x+other.x, self.y+other.y, self.z+other.z)
    def __mul__(self, scalar):
        return Point(scalar*self.x, scalar*self.y, scalar*self.z)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    def __truediv__(self, scalar):
        return Point(self.x/scalar, self.y/scalar, self.z/scalar)
    def __rtruediv__(self, scalar):
        return Point(scalar/self.x, scalar/self.y, scalar/self.z)


class Field(namedtuple('Field', ['x'])):
    __slots__ = ()
    @property
    def y(self):
        return np.moveaxis(self.x, [1, 2, 3], [2, 3, 1])
    @property
    def z(self):
        return np.moveaxis(self.x, [1, 2, 3], [3, 1, 2])


def read_wavelet_integrals(base, exp, q):
    assert base in (2, 3)
    assert exp in (2, 3, 4)
    assert q == 4

    a, b = MRA._compute_or_read_cached_results(q)
    n = 2*base**exp
    s = slice(-2, 2, 1j*n)
    x, y, z = np.ogrid[s,s,s]
    mask = (x**2 + y**2 + z**2 <= 4)
    I = np.zeros((2, q, n, n, n))

    for p in range(q):
        I[0,p][mask] = np.genfromtxt(INTEGRAL_FILE % (base, exp, p, "lower"))
        I[1,p][mask] = np.genfromtxt(INTEGRAL_FILE % (base, exp, p, "upper"))

    # force symmetry
    # XXX: with base==2 and exp==4 there appear to be numerical errors near the
    # edge of the spherical domain
    I[...,n//2:,:,:] = -I[...,:n//2,:,:][...,::-1,:,:]
    #  I[...,:,n//2:,:] = I[...,:,:n//2,:][...,:,::-1,:]
    #  I[...,:,:,n//2:] = I[...,:,:,:n//2][...,:,:,::-1]

    # symmetrize last axes
    I = (I + I.swapaxes(-1, -2)) / 2.

    # I_{lower/upper}.shape == (q, n, n, n)
    I_lower = np.sum(a[:,:,None,None,None] * I[0][None,...], axis=1)
    I_upper = np.sum(b[:,:,None,None,None] * I[1][None,...], axis=1)

    return Field(ma.array(
        I_lower + I_upper,
        mask=np.broadcast_to(~mask, (q, n, n, n)),
        hard_mask=True
    ))


def _vdiff(v, a):
    s1 = [slice(None, -1)]*3
    s2 = s1.copy()
    s1[a] = slice(1, None)
    return v[tuple(s1)] - v[tuple(s2)]


def vorticity(v, dx):
    o = np.zeros([s-1 for s in v.shape[:3]]+[3])
    o[...,0] = _vdiff(v[...,2], 1) - _vdiff(v[...,1], 2)
    o[...,1] = _vdiff(v[...,0], 2) - _vdiff(v[...,2], 0)
    o[...,2] = _vdiff(v[...,1], 0) - _vdiff(v[...,0], 1)
    return o / dx


def div(v, dx):
    return sum([_vdiff(v[...,i], i) for i in range(3)]) / dx



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
