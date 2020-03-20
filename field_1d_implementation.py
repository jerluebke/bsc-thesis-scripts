#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from alpert_rokhlin_wavelets import ARWavelet, ARWaveletNumerical


def log_poisson(rs, shape):
    return 2.**(2./3.-rs.poisson(2.*np.log(2.), shape)*np.log2(1.5))


def generate_fBm(q, N, incr_seed=None, abort_at=np.inf):
    min_N = min(N, abort_at)
    num = 2**N

    A = np.random.RandomState(incr_seed).randn(q, 2**min_N-1)

    B = np.zeros(3*num)
    t = np.linspace(-1, 2, 3*num)

    for p in range(1, q+1):
        ar = ARWavelet(q, p)
        conv = ar.conv(t)
        m = 0
        for n in range(min_N):
            size = 2**(N-n)
            step = 2**n
            for k in range(step, 2*step):
                B[(k-1)*size:(k+2)*size] += \
                        2.**(-n/3.) * conv[::step] * A[p-1,m]
                m += 1

    return B[num:2*num]


def generate_fBm_intermittent(q, N, incr_seed=None, cascade_seed=None,
                              abort_at=np.inf, return_extra=False):
    min_N = min(N, abort_at)
    num = 2**N

    A = np.random.RandomState(incr_seed).randn(q, 2**min_N-1)
    W = log_poisson(np.random.RandomState(cascade_seed), 2**min_N)
    for n in range(1, min_N):
        for k in range(2**n):
            W[2**n+k] *= W[2**(n-1)+k//2]

    B = np.zeros(3*num)
    t = np.linspace(-1, 2, 3*num)

    for p in range(1, q+1):
        ar = ARWavelet(q, p)
        conv = ar.conv(t)
        m = 0
        for n in range(min_N):
            size = 2**(N-n)
            step = 2**n
            for k in range(step, 2*step):
                B[(k-1)*size:(k+2)*size] += \
                        2.**(-n/3.) * conv[::step] * A[p-1,m] * W[m]**(1/3)
                m += 1

    return (B[num:2*num], W[-2**min_N//2:]) if return_extra else B[num:2*num]


def generate_strain(q, N, incr_seed=None, cascade_seed=None,
                    abort_at=np.inf, return_extra=False):
    min_N = min(N, abort_at)
    num = 2**N

    A = np.random.RandomState(incr_seed).randn(q, 2**min_N-1)
    W = log_poisson(np.random.RandomState(cascade_seed), 2**min_N)

    for n in range(1, min_N):
        for k in range(2**n):
            W[2**n+k] *= W[2**(n-1)+k//2]

    S = np.zeros(3*num)

    for p in range(1, q+1):
        ar = ARWaveletNumerical(q, p, num)
        conv = ar.conv_num_strain()
        m = 0
        for n in range(min_N):
            size = 2**(N-n)
            step = 2**n
            for k in range(step, 2*step):
                S[(k-1)*size:(k+2)*size] += \
                        2.**(-n/3.) * conv[::step] * A[p-1,m] * W[m]**(1/3)
                m += 1

    return (S[num:2*num+1], A, W) if return_extra else S[num:2*num]


def generate_strained_field(q, N, tau, incr_seed=None, cascade_seed=None,
                            abort_at=np.inf, return_extra=False):
    min_N = min(N, abort_at)
    num = 2**N

    S, A, W = generate_strain(q, N, incr_seed, cascade_seed, abort_at, True)
    expS = np.exp(tau*(.5-1./3.)*S)
    B = np.zeros(3*num)
    t = np.linspace(0, 1, num+1)

    for p in range(1, q+1):
        ar = ARWaveletNumerical(q, p, num)
        m = 0
        for n in range(min_N):
            size = 2**(N-n)
            step = 2**n
            for j, i in enumerate(range(step, 2*step)):
                conv = ar.conv_num(t, weight=expS[j*size:(j+1)*size+1])
                B[(i-1)*size:(i+2)*size] += \
                        2.**(-n/3.) * conv[::step] * A[p-1,m] * W[m]**(1/3)
                m += 1

    return (B[num:2*num], S, W) if return_extra else B[num:2*num]



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
