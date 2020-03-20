#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mayavi import mlab

from field_precomp import SphericalField as SphericalFieldRegular
from field_precomp_cascade import SphericalField as SphericalFieldCascade
from field_util_precomp import read_wavelet_integrals, vorticity, div


def plot_field(sf, level, **kwds):
    s = slice(sf.r-sf.r//2, sf.r+sf.r//2)
    v = sf.compute(level, **kwds)[s,s,s]
    o_mag = np.sqrt(np.sum(vorticity(v, 2./v.shape[0])**2, axis=-1))
    dv = div(v, 2./v.shape[0])

    mlab.figure()
    mlab.quiver3d(v[...,0], v[...,1], v[...,2],
                  mode='arrow', mask_points=128, scale_factor=16)
    mlab.contour3d(o_mag)
    mlab.volume_slice(o_mag)
    mlab.volume_slice(dv)
    #  mlab.savefig('vorticity-volume-slice-b2.png', size=(2000, 2000))


def plot_vortex():
    field = read_wavelet_integrals(2, 4, 4)
    noise = np.random.randn(3, 4, 1, 1, 1)
    noise_field = np.ma.sum([
        noise[2]*field.y - noise[1]*field.z,
        noise[0]*field.z - noise[2]*field.x,
        noise[1]*field.x - noise[0]*field.y
    ], axis=1)
    dnf = div(np.moveaxis(noise_field, 0, -1), 2./32.)

    mlab.figure()
    mlab.quiver3d(noise_field[0], noise_field[1], noise_field[2],
                  mode='arrow')
    mlab.volume_slice(dnf)


print('if running from IPython, run `%gui qt`')

plot_field(SphericalFieldRegular(4, 7, 2, random_rotation=False), 3)
plot_field(SphericalFieldRegular(2, 4, 3, random_rotation=True), 3)
plot_field(SphericalFieldCascade(2, 4, 3, random_rotation=True), 4,
           return_cascade=False)
plot_vortex()



# Checking different quadratures for the unit ball
def check_quadpy_schemes():
    import quadpy
    from mra_sympy import MRA
    schemes = [getattr(quadpy.ball, s)() for s in quadpy.ball.__all__]
    a, b = MRA._compute_or_read_cached_results(4)

    n = 2*2**4
    s = slice(-2, 2, 1j*n)
    X, Y, Z = np.mgrid[s,s,s]
    domain = np.stack((X, Y, Z), axis=0)

    def psi(r, p):
        return np.piecewise(
            r, [(0.<r)&(r<.5), (.5<r)&(r<1.)],\
            [np.poly1d(a[p][::-1]), np.poly1d(b[p][::-1]), 0.]
        )

    def func(x, y, h, p):
        x = np.expand_dims(x, (1, 2, 3))
        y = np.expand_dims(y, -1)
        r = np.sqrt(np.sum(x**2, axis=0))
        return (x-y) / np.sum((x-y)**2, axis=0)**(h/2) * psi(r, p)

    for p in range(4):
        for scheme in schemes:
            I = scheme.integrate(lambda x: func(x, domain, 13/6, p), [0., 0., 0.], 1.)
            o = vorticity(np.moveaxis(I, 0, -1), 2/n)
            print('[%d] -- %s\t: %f' % (p, scheme.name, np.max(np.abs(o))))


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
