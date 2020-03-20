#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import numpy.ma as ma
from scipy.ndimage import zoom as _ni_zoom
from scipy.spatial.transform import Rotation

from field_util_precomp import read_wavelet_integrals, Field, Point


H = 1/3
FOUR_PI = 4.*math.pi


class SphericalField:
    # divide by sqrt(2) to normalize edge length to one
    # keep center (0,0,0)
    CUBOCTAHEDRON_VERTICES = np.array([
        ( 1,  1,  0),
        (-1,  1,  0),
        ( 1, -1,  0),
        (-1, -1,  0),
        ( 1,  0,  1),
        (-1,  0,  1),
        ( 1,  0, -1),
        (-1,  0, -1),
        ( 0,  1,  1),
        ( 0, -1,  1),
        ( 0,  1, -1),
        ( 0, -1, -1),
        ( 0,  0,  0)
    ]) / math.sqrt(2)

    # divide by sqrt(3) to normalize diagonal length to one
    CUBE_VERTICES = np.array([
        ( 1,  1,  1),
        ( 1,  1, -1),
        ( 1, -1,  1),
        ( 1, -1, -1),
        (-1,  1,  1),
        (-1,  1, -1),
        (-1, -1,  1),
        (-1, -1, -1)
    ]) / math.sqrt(3)

    def __init__(self, Nc, Np, base, q=4, random_rotation=False,
                 noise_seed=None, rotation_seed=None, cascade_seed=None):
        # b**Nc: resolution of single field component
        # b**Np: resolution of result
        # q: wavelet order

        # basic building block of the field
        self.field = read_wavelet_integrals(base, Nc, q)

        # base, initial radius, component radius, initial zoom factor, number
        # of grid points
        self.b = base
        self.r = base**Np
        self.rc = base**Nc
        self.z = base**(Np-Nc)
        self.num = 2*self.r

        self.random_rotation = random_rotation
        self.vertices = {
            2 : self.CUBE_VERTICES,
            3 : self.CUBOCTAHEDRON_VERTICES
        }.get(base)

        # save wavelet order for noise generation
        self.q = q

        # RandomState instances
        self.noise_rs = np.random.RandomState(noise_seed)
        self.rotation_rs = np.random.RandomState(rotation_seed)
        self.cascade_rs = np.random.RandomState(cascade_seed)


    def compute(self, levels, return_cascade=False):
        radius = self.r
        z = self.z

        # result, velocity and cascade
        v = np.zeros((self.num, self.num, self.num, 3))
        c = np.ones((self.num, self.num, self.num))

        # center of initial sphere
        points = [Point(radius, radius, radius)]

        # start one level higher to fill the whole domain
        radius *= self.b
        z *= self.b

        for n in range(levels):
            fs, vs = self._field_and_domain_bounds(points, min(self.r, radius))

            # noises.shape == (len(points), 3, q, 1, 1, 1)
            noises = self._make_noise(len(points))[...,None,None,None]
            # interp_field.shape == (3, q, 2*r, 2*r, 2*r)
            interp_field = self._interpolate(z)

            w = self._make_multiplier(len(points))

            for i in range(len(points)):
                # noise_field.shape == (3, 2*r, 2*r, 2*r)
                noise_field = ma.sum([
                    noises[i,2]*interp_field.y - noises[i,1]*interp_field.z,
                    noises[i,0]*interp_field.z - noises[i,2]*interp_field.x,
                    noises[i,1]*interp_field.x - noises[i,0]*interp_field.y,
                ], axis=1)[(...,*fs[i])]

                c[vs[i]][~noise_field[0].mask] *= w[i]**(1./3.)
                noise_field *= c[(None,*vs[i])]
                noise_field = np.moveaxis(noise_field, 0, -1)

                v[(*vs[i],...)][~noise_field.mask] += \
                        self.b**(-n*H) * noise_field[~noise_field.mask]

            z /= self.b
            radius //= self.b
            points = self._subdivide_sphere(points, radius//2)

        # Biot-Savart: -1/(4 pi)
        v = -v / FOUR_PI
        return (v, c) if return_cascade else v


    def _field_and_domain_bounds(self, points, radius):
        # field component bound functions (whole sphere)
        lower = lambda p: 0 if p-radius > 0 else radius-p
        upper = lambda p: 2*radius if p+radius < self.num else radius+self.num-p

        fs = []
        vs = []

        for point in points:
            fs.append(tuple((
                slice(lower(point.x), upper(point.x)),
                slice(lower(point.y), upper(point.y)),
                slice(lower(point.z), upper(point.z)),
            )))
            vs.append(tuple((
                slice(max(point.x-radius, 0), min(point.x+radius, self.num)),
                slice(max(point.y-radius, 0), min(point.y+radius, self.num)),
                slice(max(point.z-radius, 0), min(point.z+radius, self.num)),
            )))

        return fs, vs


    def _make_noise(self, num):
        return ma.asarray(self.noise_rs.randn(num, 3, self.q))


    def _make_multiplier(self, num):
        return self.b**(2./3. - np.log(1.5)/np.log(self.b) \
                        * self.cascade_rs.poisson(2.*np.log(self.b), size=num))


    def _interpolate(self, z):
        if z > 1:
            bound = slice(None, None) if z*self.rc < self.r \
                    else slice(int(self.rc-self.r//z), int(self.rc+self.r//z))
            return Field(self._zoom(self.field.x[...,bound,bound,bound],
                                    (1, z, z, z)))

        elif z < 1:
            step = int(1./z)
            return Field(self.field.x[...,::step,::step,::step])

        else:
            return self.field


    def _subdivide_sphere(self, points, radius):
        new_points = []
        vertices = radius * self.vertices

        for point in points:
            if self.random_rotation:
                vertices = Rotation.random(random_state=self.rotation_rs).apply(vertices)

            for vertex in vertices:
                new_points.append(point + vertex)

        return new_points


    @staticmethod
    def _zoom(a, z):
        out = ma.zeros(tuple([int(round(ii*zz)) for ii, zz in zip(a.shape, z)]))
        out = _ni_zoom(a.data, z, order=1, mode='nearest', output=out)
        mask = _ni_zoom(a.mask, z, order=0, mode='constant', cval=True)
        out.mask = mask
        return out



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
