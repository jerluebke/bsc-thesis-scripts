#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from physt import h1

#  import dill; dill.settings['recurse'] = True
from pathos.multiprocessing import ProcessPool as Pool

from field_precomp import SphericalField as SphericalFieldRegular
from field_precomp_cascade import SphericalField as SphericalFieldCascade


TO_INCH = 1./2.54
NNODES = 6
#  DATA_PATH = '/home/jel/Documents'
DATA_PATH = '.'
FNAME = os.path.join(DATA_PATH, 'data/stat_3d_b%d_%s_%%s.np%%s')


def increments(fname, samples, base, Field, **kwds):
    assert base in (2, 3)

    if base == 2:
        sf = Field(3, 7, 2, **kwds)
        levels = 5
    elif base == 3:
        sf = Field(2, 4, 3, **kwds)
        levels = 3

    s = slice(sf.r-sf.r//(2*base), sf.r+sf.r//(2*base))
    num = 2*(sf.r//(2*base))

    def worker(*unused):
        return sf.compute(levels)[s,s,s,:]

    data = np.lib.format.open_memmap(fname % ('data', 'y'), mode='w+', dtype=np.float64,
                                     shape=(samples, num, num, num, 3))

    with Pool(NNODES) as p:
        # shape = (samples, num, num, num, 3)
        res = p.uimap(worker, range(samples))
        for i, field in enumerate(res):
            data[i] = field


def compute_stats(fname):
    data = np.load(fname % ('data', 'y'), mmap_mode='r')

    num = data.shape[1]
    d_arr = np.arange(1, num-1, 1)
    #  p_arr = np.arange(1, 13)
    p_arr = np.array([1, 2])

    # energy spectrum
    E = np.zeros((num//2-1, 3))
    f = np.fft.fftfreq(num, 2./num)[1:num//2]
    for i in range(3):
        reshaped_data = np.reshape(np.moveaxis(data, 1+i, 0), (num, -1))
        cov = np.mean(np.cov(reshaped_data, bias=True), axis=-1)
        E[:,i] = np.abs(np.fft.fft(cov)[1:num//2])

    np.savez(fname % ('energy', 'z'), f=f, E=E)
    del f; del E

    # increment moments and pdf
    moments = np.zeros((d_arr.size, 3, p_arr.size))
    pdfs = np.zeros((d_arr.size, 100, 2, 3))
    for i in range(d_arr.size):
        d = d_arr[i]
        ix = data[:,d:,...,0] - data[:,:-d,...,0]
        iy = data[:,:,d:,:,1] - data[:,:,:-d,:,1]
        iz = data[:,...,d:,2] - data[:,...,:-d,2]
        for j, i_arr in enumerate([ix, iy, iz]):
            h = h1(i_arr/np.expand_dims(i_arr.std(axis=1+j), 1+j), bins=100)
            h.normalize(True)
            pdfs[i,:,0,j] = h.bin_centers
            pdfs[i,:,1,j] = h.frequencies
            for k in range(p_arr.size):
                moments[i,j,k] = np.mean(np.mean(np.abs(i_arr)**p_arr[k], axis=1+j))

    np.savez(fname % ('moments', 'z'), d=d_arr, m=moments, p=pdfs)
    del d_arr; del moments; del pdfs


def compute_pdfs(fname, d_arr):
    data = np.load(fname % ('data', 'y'), mmap_mode='r')
    pdfs = np.zeros((d_arr.size, 3, 2, 100))
    for i in range(d_arr.size):
        d = d_arr[i]
        ix = data[:,d:,...,0] - data[:,:-d,...,0]
        iy = data[:,:,d:,:,1] - data[:,:,:-d,:,1]
        iz = data[:,...,d:,2] - data[:,...,:-d,2]
        for j, i_arr in enumerate([ix, iy, iz]):
            tmp = i_arr / i_arr.std()
            h = h1(tmp[np.abs(tmp)<20], bins=100)
            h.normalize(True)
            pdfs[i,j,0] = h.bin_centers
            pdfs[i,j,1] = h.densities
    np.save(fname % ('pdfs', 'y'), pdfs)



def plot_spectra():
    #  colors = ['tab:blue', 'tab:orange', 'tab:green']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ls = ['-', '--']
    base = [2, 3]
    what = ['regular', 'cascade']
    offset = [0.01, 0.001]

    fig, axes = plt.subplots(2, 1, figsize=(11*TO_INCH, 13*TO_INCH), dpi=300)
    for k in range(2):
        for j in range(2):
            data = np.load((FNAME % (base[k], what[j])) % ('energy', 'z'))
            f, E = data['f'], data['E']
            for i in range(3):
                axes[k].plot(f, E[:,i], c=colors[i], ls=ls[j])
                axes[k].plot(f, offset[k]*f**(-5/3), 'k-.')
        axes[k].set_xscale('log')
        axes[k].set_yscale('log')
        axes[k].set_ylabel('$E(k)$')
        axes[k].legend([
                plt.Line2D([], [], ls='-', c=colors[0]),
                plt.Line2D([], [], ls='-', c=colors[1]),
                plt.Line2D([], [], ls='-', c=colors[2]),
                plt.Line2D([], [], ls=ls[0], c='k'),
                plt.Line2D([], [], ls=ls[1], c='k'),
            ], [
                '$x$', '$y$', '$z$', 'regular', 'cascade',
            ],
            #  loc='upper right',
            loc='lower left',
            ncol=5,
            handlelength=1,
            columnspacing=1,
        )
    axes[-1].set_xlabel('$k$')
    plt.tight_layout(pad=.27)
    plt.savefig('stats/energy-3d.pdf', dpi='figure')


def plot_covariance():
    #  colors = ['tab:blue', 'tab:orange', 'tab:green']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ls = ['-', '--']
    base = [2, 3]
    size = [64, 40]
    what = ['regular', 'cascade']

    fig, axes = plt.subplots(2, 1, figsize=(11*TO_INCH, 13*TO_INCH), dpi=300)
    for k in range(2):
        for j in range(2):
            data = np.load((FNAME % (base[k], what[j])) % ('moments', 'z'))
            d, m = data['d'] / size[k], data['m']
            for i in range(3):
                axes[k].plot(d, m[:,i,1], c=colors[i], ls=ls[j])
                axes[k].plot(d[d>7e-2], 4e-3*d[d>7e-2]**(2/3), 'k-.')
                axes[k].plot(d[d<1e-1], 1e-1*d[d<1e-1]**2, 'k:')
        axes[k].set_xscale('log')
        axes[k].set_yscale('log')
        axes[k].set_ylabel('$S_2(l)$')
        axes[k].legend([
                plt.Line2D([], [], ls='-', c=colors[0]),
                plt.Line2D([], [], ls='-', c=colors[1]),
                plt.Line2D([], [], ls='-', c=colors[2]),
                plt.Line2D([], [], ls=ls[0], c='k'),
                plt.Line2D([], [], ls=ls[1], c='k'),
            ], [
                '$x$', '$y$', '$z$', 'regular', 'cascade',
            ],
            loc='lower right',
            ncol=5,
            handlelength=1,
            columnspacing=1,
        )
    axes[-1].set_xlabel('$l$')
    plt.tight_layout(pad=.27)
    plt.savefig('stats/cov-3d.pdf', dpi='figure')


def plot_pdfs():
    #  colors = plt.get_cmap('tab10').colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    base = [2, 3]
    kind = ['regular', 'cascade']
    d_list = [[16, 8, 4, 2], [12, 9, 6, 3]]
    x_arr = np.linspace(-3, 3, 100)
    which = 'abcd'
    j = 0

    fig, axes = plt.subplots(2, 2, figsize=(11*TO_INCH, 11*TO_INCH), dpi=300,
                             sharex=True, sharey=True)
    for b in range(2):
        for k in range(2):
            #  p = np.load((FNAME % (base[b], kind[k])) % ('moments', 'z'))['p']
            p = np.load((FNAME % (base[b], kind[k])) % ('pdfs', 'y'))
            for i, d in enumerate(d_list[b]):
                axes[b,k].plot(p[i,0,0], i+np.log10(p[i,0,1]), c=colors[i], ls='-')
                axes[b,k].plot(p[i,1,0], i+np.log10(p[i,1,1]), c=colors[i], ls='--')
                axes[b,k].plot(p[i,2,0], i+np.log10(p[i,2,1]), c=colors[i], ls=':')
            axes[b,k].plot(x_arr, np.log(np.exp(-x_arr**2/4.5)/np.sqrt(2.*np.pi)), 'k-.')
            axes[b,k].text(.05, .95, r'$(%s)$' % which[j], transform=axes[b,k].transAxes,
                           va='top', ha='left')
            axes[b,k].set_xlim(-6, 6)
            j += 1

    for ax in axes[-1,:]:
        ax.set_xlabel(r'$\delta{u}_l$')
    for ax in axes[:,0]:
        ax.set_ylabel('pdf (a.u.)')

    axes.flat[-1].legend(
        [plt.Line2D([], [], c='k', ls='-'),
         plt.Line2D([], [], c='k', ls='--'),
         plt.Line2D([], [], c='k', ls=':')],
        ['$x$', '$y$', '$z$'],
        loc='lower right',
        ncol=3,
        handlelength=1,
        columnspacing=1,
    )

    plt.tight_layout(pad=.27)
    plt.savefig('stats/pdf-3d.pdf', dpi='figure')



if __name__ == '__main__':
    fname_list = [
        FNAME % (2, 'regular'),
        FNAME % (2, 'cascade'),
        FNAME % (3, 'regular'),
        FNAME % (3, 'cascade'),
    ]

    #  print('computing samples...')
    #  increments(fname_list[0], 200, 2, SphericalFieldRegular, random_rotation=False)
    #  increments(fname_list[1], 200, 2, SphericalFieldCascade, random_rotation=False)
    #  increments(fname_list[2], 5000, 3, SphericalFieldRegular, random_rotation=True)
    #  increments(fname_list[3], 5000, 3, SphericalFieldCascade, random_rotation=True)
    #
    #  print('computing stats...')
    #  with Pool(NNODES) as p:
    #      p.map(compute_stats, fname_list)

    #  for fname in fname_list[:2]:
    #      compute_pdfs(fname, np.array([16, 8, 4, 2]))
    #  for fname in fname_list[2:]:
    #      compute_pdfs(fname, np.array([12, 9, 6, 3]))
    #
    #  print('done.')

    plot_spectra()
    plot_covariance()
    plot_pdfs()
    #  plt.close()



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
