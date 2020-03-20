#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from physt import h1

from pathos.multiprocessing import ProcessPool as Pool
# import dill; dill.settings['recurse'] = True

from field_1d import (
    generate_fBm,
    generate_fBm_intermittent,
    generate_strain,
    generate_strained_field
)


TO_INCH = 1./2.54
NNODES = 6
#  DATA_PATH = '/home/jel/Documents'
DATA_PATH = '.'
FNAME = os.path.join(DATA_PATH, 'data/stat_1d_%s_%%s.np%%s')

FIELDS = {
    'regular'       :   generate_fBm,
    'intermittent'  :   generate_fBm_intermittent,
    #  'strain'        :   generate_strain,
    'skewed-t0'     :   lambda *args, **kwds:
                            generate_strained_field(*args, tau=.0, **kwds),
    'skewed-t1'     :   lambda *args, **kwds:
                            generate_strained_field(*args, tau=.1, **kwds),
    'skewed-t2'     :   lambda *args, **kwds:
                            generate_strained_field(*args, tau=.2, **kwds),
    'skewed-t4'     :   lambda *args, **kwds:
                            generate_strained_field(*args, tau=.4, **kwds),
}


def field_samples(N, field_func, *args, **kwds):
    f = lambda *unused: field_func(*args, **kwds)
    with Pool(NNODES) as p:
        samples = np.array(p.map(f, range(N)))
    return samples


def get_field_samples(name, N, err_on_missing=True, override=False):
    fname = (FNAME % name) % (N, 'y')
    if not override and os.path.isfile(fname):
        samples = np.load(fname, mmap_mode='r')
    elif err_on_missing:
        raise OSError('sample file missing')
    else:
        samples = field_samples(N, FIELDS[name], 4, 10, abort_at=6)
        np.save(fname, samples)
    return samples




def energy_spectrum(samples, name):
    fname = name % ('energy', 'z')
    if os.path.isfile(fname):
        data = np.load(fname)
        f, E = data['f'], data['E']
    else:
        cov = np.mean(np.cov(samples.T), axis=-1)
        E = np.abs(np.fft.fft(cov)[1:cov.size//2])
        f = np.fft.fftfreq(cov.shape[0], 1./cov.shape[0])[1:cov.size//2]
        np.savez(fname, f=f, E=E)
    return f, E


def check_stationarity(samples, incr, name):
    fname = name % ('stationarity', 'y')
    if os.path.isfile(fname):
        cov = np.load(fname)
    else:
        cov = np.mean((samples[:,incr:]-samples[:,:-incr])**2, axis=0)
        cov /= cov.mean()
        np.save(fname, cov)
    return cov


def compute_moments(samples, name):
    fname = name % ('moments', 'z')
    if os.path.isfile(fname):
        data = np.load(fname)
        displ, moments = data['d'], data['m']
    else:
        p_arr = np.arange(1, 13)
        displ = np.arange(1, samples.shape[-1]//4)
        moments = np.zeros((displ.size, p_arr.size, 2))
        for i in range(displ.size):
            incr = samples[:,displ[i]:]-samples[:,:-displ[i]]
            for j in range(p_arr.size):
                moments[i,j,0] = np.mean(np.abs(incr)**p_arr[j])
                moments[i,j,1] = np.std(np.abs(incr)**p_arr[j])
        np.savez(fname, d=displ, m=moments)

    return displ / 1024, moments


def compute_pdfs(samples, name):
    fname = name % ('pdfs', 'y')
    if os.path.isfile(fname):
        pdfs = np.load(fname)
    else:
        displ = np.array([2, 4, 8, 16, 32, 64])
        pdfs = np.zeros((displ.size, 2, 100))
        for i in range(displ.size):
            incr = samples[:,displ[i]:] - samples[:,:-displ[i]]
            incr /= incr.std()
            h = h1(incr[np.abs(incr)<20], bins=100)
            h.normalize(True)
            pdfs[i,0] = h.bin_centers
            pdfs[i,1] = h.densities
        np.save(fname, pdfs)
    return pdfs


def compute_skewness(samples, name):
    fname = name % ('skewness', 'y')
    if os.path.isfile(fname):
        skew = np.load(fname)
    else:
    #      skew = None
    #  return skew
        displ = np.arange(1, samples.shape[-1]//4)
        skew = np.zeros(displ.size)
        for i in range(displ.size):
            incr = samples[:,displ[i]:] - samples[:,:-displ[i]]
            skew[i] = np.mean(incr**3)
        np.save(fname, skew)
    return skew


def compute_slopes(d, moments):
    hi = 9e-2
    lo = 9e-3
    domain = (d>lo) & (d<hi)
    zeta = []
    for p in range(moments.shape[-1]):
        zeta.append(linregress(np.log(d[domain]),
                               np.log(moments[domain,p])).slope)
    return zeta


def fit(fname, save=False):
    from scipy import odr

    def red_chi_sq(f, B, x, y, sy):
        return np.sum((y - f(B, x))**2 / sy**2) / (len(y) - len(B))

    def r_sq(f, B, x, y):
        return 1. - np.sum((y - f(B, x))**2) / np.sum((y - y.mean())**2)

    def linear_model_func(B, x):
        return B[0]*x+B[1]

    linear_model = odr.Model(linear_model_func)

    fname = fname % ('moments', 'z')
    m_file = np.load(fname)
    m, d = m_file['m'], m_file['d']

    hi = 9e-2
    lo = 9e-3
    d = d / 1024
    domain = (lo<d) & (d<hi)

    d_m = (m[...,1] / m[...,0])[domain]
    l_m = np.log(m[...,0][domain])
    l_d = np.log(d[domain])
    zeta_arr = np.empty((m.shape[1], 2))
    C_arr = np.empty_like(zeta_arr)
    d_z_arr = np.empty(m.shape[1])
    d_C_arr = np.empty_like(d_z_arr)
    r_sq_arr = np.empty_like(d_z_arr)

    print('\n\n\t[%s]\n' % fname)
    print('[p]\tzeta\t\t\t\t\tC\t\t\t\t\tred chi sq\tr^2')
    print('===\t===\t\t\t\t\t===\t\t\t\t\t===\t\t===')
    for i in range(l_m.shape[1]):
        data = odr.RealData(l_d, l_m[:,i], sy=d_m[:,i])
        out = odr.ODR(data, linear_model, beta0=[1., 0.]).run()
        B, sd_B = out.beta, out.sd_beta
        red_chi_sq_test = red_chi_sq(
            linear_model_func, B, l_d, l_m[:,i], d_m[:,i])
        r_sq_test = r_sq(linear_model_func, B, l_d, l_m[:,i])
        zeta_arr[i,0], zeta_arr[i,1] = B[0], sd_B[0]
        C_arr[i,0], C_arr[i,1] = B[1], sd_B[1]
        d_z_arr[i] = abs(sd_B[0]/B[0])*100
        d_C_arr[i] = abs(sd_B[1]/B[1])*100
        r_sq_arr[i] = r_sq_test
        print('[%2d]\t%f +- %f (%.2f%%)\t\t%f +- %f (%.2f%%)\t\t%f\t%f' \
              % (i, B[0], sd_B[0], d_z_arr[i],
                 B[1], sd_B[1], d_C_arr[i],
                 red_chi_sq_test, r_sq_test))

    if save:
        np.savetxt(os.path.splitext(fname)[0]+'.csv',
                   np.vstack((np.arange(1, 13), zeta_arr[:,0], zeta_arr[:,1],
                              d_z_arr, C_arr[:,0], C_arr[:,1], d_C_arr,
                              r_sq_arr)).T,
                   fmt=['%d']+7*['%.3f'],
                   header='p,t,dt,dtr,c,dc,dcr,rsq',
                   comments='', delimiter=',')

    return (zeta_arr, C_arr), (l_d, l_m)




def plot_moments(fname):
    (z, C), (d, m) = fit(fname)

    plt.figure()
    for i in range(m.shape[1]):
        plt.plot(d, m[:,i])
        plt.plot(d, z[i,0]*d+C[i,0], 'k--')


def plot_pdfs(field_list, nrows, ncols, name):
    d_list = [64, 32, 16, 8, 4, 2]
    x_arr = np.linspace(-5, 5, 100)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                          squeeze=False,
                          figsize=(ncols*5.5*TO_INCH, nrows*5.5*TO_INCH), dpi=300)
    for ax, field, which in zip(axes.flat, field_list, 'abcdef'):
        fname = FNAME % field
        p = compute_pdfs(None, fname)
        for i, j in enumerate(d_list):
            ax.plot(p[-(i+1),0], i+np.log10(p[-(i+1),1]), label=r'$l=%d\mathrm{d}x$' % j)
        ax.plot(x_arr, -.5+np.log(np.exp(-x_arr**2/4.5)/np.sqrt(2.*np.pi)), 'k--',
                label='std normal')
        ax.text(.05, .95, r'$(%s)$' % which, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left')
        ax.set_xlim(-11, 11)

    for ax in axes[-1,:]:
        ax.set_xlabel(r'$\delta u_l$')
    for ax in axes[:,0]:
        ax.set_ylabel('pdf (a.u.)')
    #  axes.flat[-1].legend()
    fig.subplots_adjust(left=.01)
    plt.tight_layout(h_pad=.27, w_pad=.27)
    plt.savefig('stats/pdfs-%s-1d.pdf' % name, dpi='figure')


def plot_spectrum(field_list, label_list, color_list, offset):
    fig, axes = plt.subplots(2, 1, figsize=(11*TO_INCH, 13*TO_INCH), dpi=300,
                             sharex=True)

    for i in range(2):
        for field, label, color in zip(field_list[i], label_list[i],
                                       color_list[i]):
            name = FNAME % field
            f, E = energy_spectrum(None, name)
            axes[i].plot(f[32:], E[32:], label=label, color=color)
        axes[i].plot(f[32:], offset[i]*f[32:]**(-5/3), 'k--')
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_ylabel(r'$E(k)$')
        axes[i].legend(loc='lower left', ncol=2)
    axes.flat[-1].set_xlabel(r'$k$')
    plt.tight_layout(pad=.27)
    plt.savefig('stats/spectrum-1d.pdf', dpi='figure')


def plot_covariance(field_list, label_list, offset1, offset2):
    fig, axes = plt.subplots(2, 1, figsize=(11*TO_INCH, 13*TO_INCH), dpi=300,
                             sharex=True)

    for i in range(2):
        for field, label in zip(field_list[i], label_list[i]):
            name = FNAME % field
            d, m = compute_moments(None, name)
            axes[i].plot(d, m[:,1,0], label=label)
        axes[i].plot(d, offset1[i]*d**(2/3), 'k--')
        axes[i].plot(d[d<6e-3], offset2[i]*d[d<6e-3]**2, 'k:')
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_ylabel(r'$S_2(l)$')
        axes[i].legend(loc='lower right')
    axes.flat[-1].set_xlabel(r'$l$')
    plt.tight_layout(pad=.27)
    plt.savefig('stats/cov-1d.pdf', dpi='figure')


def plot_S12(field_list, label_list):
    plt.figure(figsize=(11*TO_INCH, 6*TO_INCH), dpi=300)
    plt.loglog()
    for field, label in zip(field_list, label_list):
        name = FNAME % field
        d, m = compute_moments(None, name)
        plt.plot(d, m[:,11,0]/np.mean(m[:,11,0]), label=label)
    z = 12/9+2-(2/3)**(12/3)
    plt.plot(d, d**4/np.mean(d**4), 'k--', label='K41')
    plt.plot(d, d**z/np.mean(d**z), 'k-.', label='log-Poisson')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$S_{12}(l)~\mathrm{(a.u.)}$')
    plt.legend()
    plt.tight_layout(pad=.27)
    plt.savefig('stats/s12-1d.pdf', dpi='figure')


def plot_stationarity(field_list, label_list, loc_list):
    fig, axes = plt.subplots(3, 1, figsize=(11*TO_INCH, 13*TO_INCH), dpi=300,
                             sharex=True)

    for i in range(3):
        for field, label in zip(field_list[i], label_list[i]):
            name = FNAME % field
            s = check_stationarity(None, None, name)
            t = np.linspace(0, 1, s.size)
            axes[i].plot(t, s, label=label)
        axes[i].plot([0, 1], [1, 1], 'k--')
        axes[i].legend(loc=loc_list[i])

    #  axes[0].set_ylim(.93, 1.07)
    axes[2].set_xlim(0, 1)
    axes[1].set_ylabel(r'$\langle\delta{u}_l^2(t)\rangle/\mathsf{E}_t[\langle\delta{u}_l^2\rangle]$')
    #  axes[1].set_ylabel(r'$\tilde{S}_2(l,t)/\mathsf{E}_t[\tilde{S}_2(l,t)]$')
    axes[2].set_xlabel(r'$t$')
    plt.tight_layout(pad=.27)
    plt.savefig('stats/stat-1d.pdf', dpi='figure')


def plot_zetas(field_list,
               label_list,
               marker_list=['o', 's', 'v', '^', '<', '>']):

    p_arr = np.arange(1, 13)
    def zeta_SL(p):
        return p/9 + 2 - 2*(2/3)**(p/3)

    plt.figure(figsize=(11*TO_INCH, 10*TO_INCH), dpi=300)

    for field, label, marker in zip(field_list, label_list, marker_list):
        #  d, m = compute_moments(None, FNAME % name)
        #  zeta = compute_slopes(d, m)
        zeta, _ = fit(FNAME % field)[0]
        plt.errorbar(p_arr, zeta[:,0], yerr=zeta[:,1], marker=marker, linestyle='none', label=label)

    plt.plot(p_arr, p_arr/3, 'k--',
             label=r'$\mathrm{K41}$')
    plt.plot(p_arr, zeta_SL(p_arr), 'k-.',
             label=r'$\mathrm{log}$-$\mathrm{Poisson}$')
    plt.xlabel(r'$p$')
    plt.ylabel(r'$\zeta_p$')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=.27)
    plt.savefig('stats/zeta-1d.pdf', dpi='figure')


def plot_skew(field_list):
    d = np.arange(1, 256) / 1024
    fig, axes = plt.subplots(4, 1, sharex=True,
                           figsize=(11*TO_INCH, 13*TO_INCH), dpi=300)
    for ax, field, which in zip(axes, field_list, 'abcd'):
        name = FNAME % field
        s = compute_skewness(None, name)
        ax.plot(d, s)
        ax.text(.05, .95, r'$(%s)$' % which, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left')
    axes.flat[-1].set_xlabel(r'$l$')
    fig.subplots_adjust(left=.01)
    fig.text(.02, .55, r'$S_3(l)$', rotation='vertical', ha='center', va='center')
    plt.tight_layout(h_pad=.27)
    plt.savefig('stats/skewness-1d.pdf', dpi='figure')




if __name__ == '__main__':
    #  print('generating samples...')
    #  sample_list = []
    #  for field in FIELDS.keys():
    #      print('processing %s...' % field)
    #      sample_list.append(get_field_samples(field, 100_000, False))
    #
    #  def worker(*args):
    #      samples, field = args[0]
    #      name = FNAME % field
    #      energy_spectrum(samples, name)
    #      check_stationarity(samples, 10, name)
    #      compute_skewness(samples, name)
    #      compute_pdfs(samples, name)
    #      compute_moments(samples, name)
    #
    #  print('computing stats...')
    #  with Pool() as p:
    #      p.map(worker, zip(sample_list, FIELDS.keys()))
    #
    #  print('done.')

    field_list = list(FIELDS.keys())
    label_list = [r'regular', r'cascade'] \
                  + [r'$\tau=%s$' % f for f in (0,.1,.2,.4)]
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    #  color_list = plt.get_cmap('tab10').colors[:4]
    #  for field in field_list:
        #  plot_moments(FNAME % field)
        #  fit(FNAME % field, save=True)

    #  plot_pdfs(field_list[:2], 1, 2, 'regular')
    #  plot_pdfs(field_list[2:], 2, 2, 'skewed')
    #  plot_pdfs(field_list, 3, 2, 'all')
    #  plot_zetas(field_list, label_list)
    plot_S12(field_list[2:], label_list[2:])
    #  plot_skew(field_list[2:])

    #  plot_spectrum([field_list[:2], reversed(field_list[2:])],
    #                [label_list[:2], reversed(label_list[2:])],
    #                [color_list[:2], reversed(color_list)],
    #                [.3, 1.])
    plot_covariance([field_list[:2], field_list[2:]],
                    [label_list[:2], label_list[2:]],
                    [.45, .6], [6.5e2, 1e3])

    #  plot_stationarity([field_list[:2], field_list[2:-1], [field_list[-1]]],
    #                    [label_list[:2], label_list[2:-1], [label_list[-1]]],
    #                    ['lower left', 'lower left', 'upper left'])
    #  #  plot_stationarity([f+'_incr30' for f in field_list[2:-1]])
    #  plot_stationarity([field_list[-1]+'_incr30'])



#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
