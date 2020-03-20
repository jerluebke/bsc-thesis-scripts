#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from functools import reduce
from operator import mul

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.special import hyp2f1


COEFFS = [
    [[1]],
    [[-1, 2],
     [-2, 3]],
    [[1, -24, 30],
     [3, -16, 15],
     [4, -15, 12]],
    [[1, 4, -30, 28],
     [-4, 105, -300, 210],
     [-5, 48, -105, 64],
     [-16, 105, -192, 105]],
    [[1, 30, 210, -840, 630],
     [-5, -144, 1155, -2240, 1260],
     [22, -735, 3504, -5460, 2700],
     [35, -512, 1890, -2560, 1155],
     [32, -315, 960, -1155, 480]],
]


def binom(a, k):
    return reduce(mul, ((a+1-i)/i for i in range(1, k+1)), 1.)



class ARWavelet:

    def __init__(self, q, p):
        assert q<6, 'q > 5 not implemented'
        assert p<=q, 'p must be smaller than q'

        self.q = q
        self.symmetry = (-1)**(q+p-1)
        self.coeffs = COEFFS[q-1][p-1]
        self.norm = math.sqrt(quad(np.poly1d(self.coeffs[::-1])**2, 0, 1)[0])


    def as_piecewise_poly(self):
        # without normalization!
        shift = np.poly1d([2, -1])
        w_poly = np.poly1d(self.coeffs[::-1])
        return [self.symmetry*w_poly(-shift), w_poly(shift)]

    def moment(self, m):
        return 2.**(-m-1.) / self.norm * np.sum([
            self.coeffs[i] * np.sum([
                binom(i, j) * (-1.)**j / (m+j+1.) \
                * (self.symmetry + (-1.)**i * (2.**(j+m+1.) - 1.))
                for j in range(i+1)
            ])
            for i in range(self.q)
        ])

    def moment_quad(self, m):
        mom_coeffs = np.zeros(m+1)
        mom_coeffs[0] = 1.
        mom_poly = np.poly1d(mom_coeffs)
        w_poly = np.poly1d(self.coeffs[::-1])
        return quad(lambda x: self.symmetry*w_poly(1-2*x)*mom_poly(x)/self.norm, 0, .5)[0] \
                + quad(lambda x: w_poly(2*x-1)*mom_poly(x)/self.norm, .5, 1)[0]

    def moment_polyint(self, m):
        w_left, w_right = self.as_piecewise_poly()
        mom_coeffs = np.zeros(m+1)
        mom_coeffs[0] = 1.

        # piecewiese polynomial scalar product with weight 1.
        P_left = np.polyint(w_left*mom_coeffs)
        P_right = np.polyint(w_right*mom_coeffs)
        return (P_right(1.) - P_right(.5)) + (P_left(.5) - P_left(0.)) / self.norm


    def conv(self, t, n=0., k=0., H=1./3.):
        #  t = 2.**n * t - k
        return 2.**(-n*H) * np.piecewise(
            t, [t>0, t<=0],
            [lambda t: self._conv_exact(t, H),
             lambda t: self.symmetry * self._conv_exact(1-t, H)]
        )

    def _conv_exact(self, t, H):
        # _conv_exact(t, H) for t>0
        # _conv_exact(1-t, H) for t<0
        h = .5 - H
        #  h = 3/2 - H
        return np.sum([
            binom(i, j) * self.coeffs[i] * (-2.)**j * (
                self.symmetry * self._mono_conv_int(t, j, h, 0, .5)
                + (-1)**i * self._mono_conv_int(t, j, h, .5, 1)
            )
            for i in range(self.q)
            for j in range(i+1)
        ], axis=0) / self.norm

    def _mono_conv_int(self, t, n, h, a, b):
        return np.piecewise(
            t, [(a<=t)&(t<=b), t<a, t>b],
            [lambda t: self._mci_larger(t, n, h, a, t) + self._mci_smaller(t, n, h, t, b),
             lambda t: self._mci_smaller(t, n, h, a, b),
             lambda t: self._mci_larger(t, n, h, a, b)]
        )

    def _mci_larger(self, t, j, h, a, b):
        # t>b>a: int_a^b (t-s)^(-h) s^n ds
        res = b**(j+1.) * hyp2f1(h, j+1., j+2., b/t)
        if isinstance(a, np.ndarray) or a != 0:
            res -= a**(j+1.) * hyp2f1(h, j+1., j+2., a/t)
        return t**(-h) * res / (j+1.)

    def _mci_smaller(self, t, j, h, a, b):
        # t<a<b: int_a^b (s-t)^(-h) s^n ds
        res = b**(j+1.-h) * hyp2f1(h, h-j-1., h-j, t/b)
        if isinstance(a, np.ndarray) or a != 0:
            res -= a**(j+1.-h) * hyp2f1(h, h-j-1., h-j, t/a)
        return res / (j+1.-h)


    def _conv_exact_naive(self, t, H):
        # looks right, but does not agree with numerical results...
        return np.sum([
            binom(i, j) * binom(j, k) * self.coeffs[i] * (-2.)**j * t**(j-k) / (k+H+.5) \
            * (self.symmetry * ((.5-t)**(k+1) * np.abs(.5-t)**(H-.5)
                               - np.sign(.5-t) * (-t)**k * np.abs(t)**(H+.5))
              + (-1.)**i * ((1.-t)**(k+1) * np.abs(1.-t)**(H-.5)
                           - np.sign(1.-t) * (.5-t)**k * np.abs(.5-t)**(H+.5)))
            for i in range(self.q)
            for j in range(i+1)
            for k in range(j+1)
        ], axis=0) / self.norm



class ARWaveletNumerical(ARWavelet):

    def __init__(self, q, p, num):
        self.p = p
        super().__init__(q, p)

        assert ~(num & (num-1)), 'num must be a power of two'
        self.num = num
        #  self.t = np.linspace(-1, 1, 2*self.num+1)
        self.t = np.linspace(-2, 2, 4*self.num-1)
        t_half = np.linspace(0, 1, num//2)

        f = self.coeffs[-1] + t_half*0
        for i in range(2, len(self.coeffs)+1):
            f = self.coeffs[-i] + f*t_half

        self.wavelet = np.zeros(num)
        self.wavelet[:num//2] = self.symmetry * f[::-1] / self.norm
        self.wavelet[num//2:] = f / self.norm


    def __call__(self, n=0., k=0.):
        return (2.**(-n)*(np.linspace(0, 1, self.num)+k), 2.**(-n/2.)*self.wavelet)


    def _norm(self, t):
        return np.sqrt(t**2+1./self.num**2)


    def conv_num(self, t, weight, H=1./3.):
        iw = interp1d(t[::t.size//(weight.size-1)], weight)(t[:-1])
        ker = self._norm(self.t)**(H-.5)
        w = self.wavelet*iw
        c = convolve(ker, w, mode='valid') / self.num
        c[:c.size//2] -= c[0]
        c[c.size//2:] -= c[-1]
        return c


    def conv_num_strain(self, H=1./3., cutoff=3.):
        ker = np.sign(self.t) * self._norm(self.t)**(H-1.5)
        conv = convolve(ker, self.wavelet, mode='valid') / self.num
        conv[:conv.size//2] -= conv[0]
        conv[conv.size//2:] -= conv[-1]

        abs_conv = np.abs(conv)
        l_slice = slice(self.num-32,self.num+12)
        r_slice = slice(2*self.num-12,2*self.num+32)
        l_view = conv[l_slice]
        r_view = conv[r_slice]
        l_sign = np.sign(l_view[np.argmax(abs_conv[l_slice])])
        r_sign = np.sign(r_view[np.argmax(abs_conv[r_slice])])
        abs_conv[l_slice] = 0.
        abs_conv[r_slice] = 0.
        if self.p == 4:
            m_slice = slice(int(1.5*self.num-32),int(1.5*self.num+32))
            m_view = conv[m_slice]
            abs_conv[m_slice] = 0.
            m_sign = +1
        hh = np.max(abs_conv) / cutoff
        l_view[np.abs(l_view)>hh] = l_sign*hh
        r_view[np.abs(r_view)>hh] = r_sign*hh
        if self.p == 4:
            m_view[np.abs(m_view)>hh] = m_sign*hh

        return conv



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
