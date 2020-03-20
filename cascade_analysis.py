#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from pathos.multiprocessing import ProcessPool as Pool
import dill; dill.settings['recurse'] = True

TO_INCH = 1./2.54


def tau_SL(p):
    return 2-2*(2/3)**p-2/3*p


def log_poisson(b, *args):
    return b**(2/3-np.random.poisson(2*np.log(b))*np.log(1.5)/np.log(b))


def cascade_all_steps(b, d, n, rand_func, *args):
    N = b**(n-1)
    M = np.empty(tuple(d*[N]+[n]))
    for i in range(n):
        step = b**i
        size = b**(n-i-1)
        coords = np.dstack(np.mgrid[tuple(d*[slice(0, step)])]).reshape(-1, d)
        for coord in coords:
            loc = tuple([slice(c*size,(c+1)*size) for c in coord]+[i])
            M[loc] = rand_func(b, *args)
    return M.cumprod(axis=-1, out=M)


def record_stats(samples, p_arr, fname):
    N = 10
    base = 2
    dim = 2
    data_fname = '_cascade_data.npy'

    #  p_arr = np.expand_dims(p_arr, tuple([d for d in range(dim+1)]))
    p_arr = p_arr[None,None,None,:]
    data = np.lib.format.open_memmap(
        data_fname, mode='w+', dtype=np.float64,
        shape=(samples, N, p_arr.size)
    )

    def worker(*unused):
        M = cascade_all_steps(base, dim, N, log_poisson)[...,None]
        #  M = np.expand_dims(M, -1)
        return np.mean(M**p_arr, axis=(0, 1))

    with Pool() as p:
        res = p.uimap(worker, range(samples))
        for i, elem in enumerate(res):
            data[i] = elem

    S_arr = np.mean(data, axis=0)
    sd_S = np.std(data, axis=0)
    np.savez(fname, S=S_arr, sd_S=sd_S)
    del data
    os.remove(data_fname)


def fit(fname, p_arr=None, save=False):
    from scipy import odr

    def red_chi_sq(f, B, x, y, sy):
        return np.sum((y - f(B, x))**2 / sy**2) / (len(y) - len(B))

    def r_sq(f, B, x, y):
        return 1. - np.sum((y - f(B, x))**2) / np.sum((y - y.mean())**2)

    def linear_model_func(B, x):
        return B[0]*x+B[1]

    linear_model = odr.Model(linear_model_func)

    S_file = np.load(fname)
    S_arr, sd_S = S_file['S'], S_file['sd_S']
    n_arr = -np.arange(S_arr.shape[0])
    tau_arr = np.empty((S_arr.shape[1], 2))
    C_arr = np.empty_like(tau_arr)
    d_tau_rel = np.empty(S_arr.shape[1])
    d_C_rel = np.empty_like(d_tau_rel)
    r_sq_arr = np.empty_like(d_tau_rel)

    #  samples = 1000
    #  N = 10
    #  base = 2
    #  dim = 2
    #  delta_S = np.sqrt(samples*base**((N-1)*dim)) / S_arr
    delta_S = sd_S / S_arr # / np.sqrt(1000)

    print('[p]\ttau\t\t\t\t\tC\t\t\t\t\tred chi sq\tr^2')
    print('===\t===\t\t\t\t\t===\t\t\t\t\t===\t\t===')
    for i in range(S_arr.shape[1]):
        data = odr.RealData(n_arr, np.log2(S_arr[:,i]), sy=delta_S[:,i])
        out = odr.ODR(data, linear_model, beta0=[1., 0.]).run()
        B, sd_B = out.beta, out.sd_beta
        red_chi_sq_test = red_chi_sq(
            linear_model_func, B, n_arr, np.log2(S_arr[:,i]), delta_S[:,i])
        r_sq_test = r_sq(linear_model_func, B, n_arr, np.log2(S_arr[:,i]))
        tau_arr[i,0], tau_arr[i,1] = B[0], sd_B[0]
        C_arr[i,0], C_arr[i,1] = B[1], sd_B[1]
        d_tau_rel[i] = abs(sd_B[0]/B[0])*100
        d_C_rel[i] = abs(sd_B[1]/B[1])*100
        r_sq_arr[i] = r_sq_test
        print('[%2d]\t%f +- %f (%.2f%%)\t\t%f +- %f (%.2f%%)\t\t%f\t%f' \
              % (i, B[0], sd_B[0], d_tau_rel[i],
                 B[1], sd_B[1], d_C_rel[i],
                 red_chi_sq_test, r_sq_test))

    if save:
        np.savetxt('cascade_structure.csv',
                   np.vstack((p_arr, tau_arr[:,0], tau_arr[:,1], d_tau_rel,
                              C_arr[:,0], C_arr[:,1], d_C_rel, r_sq_arr)).T,
                   fmt=['%d']+7*['%.3f'],
                   header='p,t,dt,dtr,c,dc,dcr,rsq',
                   comments='', delimiter=',')

    return (tau_arr, C_arr), (n_arr, np.log2(S_arr), delta_S)


def plot_stats(fname, p_arr):
    #  color_list = plt.get_cmap('tab10').colors
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    p_linspace = np.linspace(0, 50, 100)
    (tau_arr, C_arr), (n_arr, S_arr, d_S) = fit(fname)

    _, axes = plt.subplots(2, 1, sharex=True,
                           figsize=(11*TO_INCH, 13*TO_INCH), dpi=300)
    for i, c in zip(range(9), color_list):
        axes[0].plot(n_arr, tau_arr[i,0]*n_arr+C_arr[i,0], color=c, ls='--')
        axes[0].errorbar(n_arr, S_arr[:,i], yerr=d_S[:,i], fmt='x', color=c,
                         #  mfc='k', mec='k',
                         ms=8,
                     label='$p=%d$' % p_arr[i])
    axes[0].set_ylabel(r'$\log_2\langle\varepsilon_l^p\rangle$')
    axes[0].legend(ncol=2, loc='upper right')

    for i, c in zip(range(9, 14), color_list):
        axes[1].plot(n_arr, tau_arr[i,0]*n_arr+C_arr[i,0], color=c, ls='--')
        axes[1].errorbar(n_arr, S_arr[:,i], yerr=d_S[:,i], fmt='x', color=c,
                         #  mfc='k', mec='k',
                         ms=8,
                     label='$p=%d$' % p_arr[i])
    axes[1].set_ylabel(r'$\log_2\langle\varepsilon_l^p\rangle$')
    axes[1].set_xlabel(r'$-n=\log_2l$')
    axes[1].legend(loc='upper right')

    plt.tight_layout(pad=.27)
    plt.savefig('cascade-structure-functions.pdf', dpi='figure')


    plt.figure(figsize=(11*TO_INCH, 6*TO_INCH), dpi=300)

    plt.subplot(121)
    plt.errorbar(p_arr[:10], tau_arr[:10,0], yerr=tau_arr[:10,1], fmt='3',
                 ms=8, mew=1.6)
    plt.plot(p_linspace[p_linspace<=10], tau_SL(p_linspace[p_linspace<=10]), 'k--')
    plt.ylabel(r'$\tau_p$')
    plt.xlabel('$p$')

    plt.subplot(122)
    plt.errorbar(p_arr, tau_arr[:,0], yerr=tau_arr[:,1], fmt='3', ms=8, mew=1.6,
                 label='recorded')
    plt.plot(p_linspace, tau_SL(p_linspace), 'k--', label='theory')
    plt.xlabel('$p$')
    plt.legend(handlelength=1.2)

    plt.tight_layout(pad=.27)
    plt.savefig('cascade-stat.pdf', dpi='figure')


if __name__ == '__main__':
    fname = 'cascade_structure.npz'
    p_arr = np.array(list(range(1, 11)) + [20, 30, 40, 50])
    #  record_stats(1000, p_arr, fname)
    #  fit(fname, p_arr, save=True)
    plot_stats(fname, p_arr)


#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
