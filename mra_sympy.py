#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from sympy import *
from sympy.abc import x

from functools import reduce
from operator import add


def _z(M):
    #  M[np.isclose(M, 0., atol=1e-10)] = 0.
    return M


class MRA:
    def __init__(self, order, domain=[0., 0.5, 1.], full=True):
        domain = [nsimplify(d) for d in domain]
        self.k = order
        self.left = [domain[0], domain[1]]
        self.right = [domain[1], domain[2]]
        self.density = np.poly1d([1., 0., 0.])
        self.density_expr = x**2
        if full:
            self.M, self.X = self._build_M_and_X()

    def moment(self, i, domain):
        return nsimplify(Integral(self.density_expr*x**i, (x, *domain)).doit())

    def inner_prod(self, u, v):
        return u @ self.X @ v.T

    def _build_L_and_R(self):
        L = zeros(self.k)
        R = zeros(self.k)
        for i in range(self.k):
            for j in range(self.k):
                L[i,j] = self.moment(i+j, self.left)
                R[i,j] = self.moment(i+j, self.right)
        return L, R

    def _build_M_and_X(self):
        L, R = self._build_L_and_R()
        L_inv = L.inv()
        M = -L_inv @ R
        X = R @ L_inv @ (R + L)
        return M, X

    def _build_v(self):
        V_l = zeros(self.k-1, self.k)
        V_r = zeros(self.k-1, self.k)
        for i in range(self.k-1):
            for j in range(self.k):
                m = self.k + i
                V_l[i,j] = self.moment(m+j, self.left)
                V_r[i,j] = self.moment(m+j, self.right)
        return V_l, V_r

    def _build_u(self):
        V_l, V_r = self._build_v()
        X_inv = self.X.inv()
        U = (V_l @ self.M + V_r) @ X_inv
        return U

    def _proj(self, v, u):
        v_dot_u = self.inner_prod(v, u)
        u_dot_u = self.inner_prod(u, u)
        return v_dot_u / u_dot_u * u

    def gs_step(self, v, U):
        u_res = v.copy()
        for i in range(U.shape[0]):
            u_res -= self._proj(u_res, U[i,:])
        return u_res

    def gram_schmidt(self, V):
        U = zeros(*V.shape)
        for i in range(V.shape[0]):
            U[i,:] = self.gs_step(V[i,:], U[:i,:])
        return U

    def compute_alpha_and_beta(self):
        U = self.gram_schmidt(self._build_u())
        beta = zeros(self.k)
        ker = (U @ self.X).nullspace()
        beta[-1,:] = reduce(add, ker).T
        beta[:-1,:] = U
        norm_matrix = beta @ self.X @ beta.T
        for i in range(self.k):
            beta[i,:] /= sqrt(norm_matrix[i,i])
        alpha = beta @ self.M.T
        return (np.array(alpha, dtype=np.float64),
                np.array(beta, dtype=np.float64))

    def check_moments(self, alpha, beta):
        M = np.full((2, self.k, 2*self.k-1), np.nan)
        for l in range(self.k):
            for m in range(self.k+l):
                mc = np.zeros(m+1)
                mc[0] = 1.
                Pl = np.polyint(self.density * np.poly1d(alpha[l][::-1]) * mc)
                Pr = np.polyint(self.density * np.poly1d(beta[l][::-1]) * mc)
                M[0,l,m] = Pl(self.left[1]) - Pl(self.left[0]) \
                        + Pr(self.right[1]) - Pr(self.right[0])
                M[1,l,m] = np.sum([alpha[l,i] * self.moment(m+i, self.left) \
                                 + beta[l,i] * self.moment(m+i, self.right)
                                 for i in range(self.k)])
        return _z(M)

    def check_orthonormality(self, alpha, beta):
        M = np.zeros((2, self.k, self.k))
        for l in range(self.k):
            for k in range(self.k):
                Pl = np.polyint(self.density \
                                * np.poly1d(alpha[l][::-1]) \
                                * np.poly1d(alpha[k][::-1]))
                Pr = np.polyint(self.density \
                                * np.poly1d(beta[l][::-1]) \
                                * np.poly1d(beta[k][::-1]))
                M[0,l,k] = Pl(self.left[1]) - Pl(self.left[0]) \
                        + Pr(self.right[1]) - Pr(self.right[0])
                M[1,l,k] = np.sum([alpha[l,i]*alpha[k,j]*self.moment(i+j, self.left) \
                                 + beta[l,i]*beta[k,j]*self.moment(i+j, self.right)
                                 for i in range(self.k) for j in range(self.k)])
        return _z(M)

    @classmethod
    def _compute_or_read_cached_results(cls, order):
        fn = 'polycoeffs-%d.csv' % order
        if os.path.isfile(fn):
            content = np.loadtxt(fn, delimiter=',')
            assert content.size == 2*order**2
            alpha, beta = content[:order], content[order:]
        else:
            mra = cls(order)
            alpha, beta = mra.compute_alpha_and_beta()
            np.savetxt(fn, np.vstack((alpha, beta)), fmt='%.12f', delimiter=',')
        return alpha, beta

    @classmethod
    def check(cls, order):
        mra = MRA(order, full=False)
        a, b = mra._compute_or_read_cached_results(order)
        print('===')
        print('Alpert wavelets of order %d' % order)
        print('---')
        print('moments:')
        print(mra.check_moments(a, b))
        print('\northogonality:')
        print(mra.check_orthonormality(a, b))
        print('===')

    @classmethod
    def plot_it(cls, order):
        import matplotlib.pyplot as plt
        plt.figure()
        l = np.linspace(0, .5)
        r = np.linspace(.5, 1)
        a, b = cls._compute_or_read_cached_results(order)
        for i, c in zip(range(order), plt.get_cmap('tab10').colors):
            plt.plot(l, np.poly1d(a[i][::-1])(l), c=c)
            plt.plot(r, np.poly1d(b[i][::-1])(r), c=c)



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
