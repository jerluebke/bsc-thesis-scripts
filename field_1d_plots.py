#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from alpert_rokhlin_wavelets import ARWavelet, ARWaveletNumerical
from field_1d import (
    generate_fBm,
    generate_fBm_intermittent,
    generate_strain,
    generate_strained_field
)


TO_INCH = 1./2.54


plt.figure(figsize=(12*TO_INCH, 6*TO_INCH), dpi=300)
t = np.linspace(-.5, 1.5, 1024)
for p in range(1, 5):
    a = ARWavelet(4, p)
    c = a.conv(t)
    plt.plot(t, c, label='$p=%d$' % p)
    print(c[0], c[-1])
plt.xlim(-.5, 1.5)
plt.xlabel(r'$t$')
plt.ylabel(r'$\Psi^{(pq)}(t)$')
plt.legend(loc='lower right')
plt.tight_layout(pad=.27)
plt.savefig('velocity-convolution.pdf', dpi='figure')


plt.figure(figsize=(12*TO_INCH, 6*TO_INCH), dpi=300)
t = np.linspace(-1, 2, 3072)
for p in range(1, 5):
    a = ARWaveletNumerical(4, p, 1024)
    c = a.conv_num_strain()
    plt.plot(t, c, label='$p=%d$' % p)
    print(c[0], c[-1])
    print(np.max(np.abs(
        a.conv(np.linspace(-1, 2, 3*1024))\
        -a.conv_num(np.linspace(0, 1, 1025), np.ones(3)))))
plt.xlim(-.5, 1.5)
plt.xlabel(r'$t$')
plt.ylabel(r'$\Phi^{(pq)}(t)$')
plt.legend(loc='lower right')
plt.tight_layout(pad=.27)
plt.savefig('strain-convolution.pdf', dpi='figure')


plt.figure(figsize=(12*TO_INCH, 6*TO_INCH), dpi=300)
t = np.linspace(0, 1, 1024)
for n in [8, 4, 3]:
    plt.plot(t, generate_fBm(4, 10, 100, n), label='$N=%d$' % n)
plt.xlim(0, 1)
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.legend()
plt.tight_layout(pad=.27)
plt.savefig('field-1d-regular.pdf', dpi='figure')


_, axes = plt.subplots(2, 1, sharex=True,
                       figsize=(12*TO_INCH, 10*TO_INCH), dpi=300)
B, M = generate_fBm_intermittent(4, 10, 100, 200, 7, True)
axes[0].plot(t, B)
axes[0].set_ylabel('$u(t)$')
axes[1].plot(
    t,
    interp1d(t[::t.size//M.size], M, kind='nearest',
             bounds_error=False, fill_value='extrapolate')(t),
)
axes[1].set_ylabel(r'$\varepsilon_l$')
axes[1].set_xlabel('$t$')
axes[1].set_xlim(0, 1)
plt.tight_layout(pad=.27)
plt.savefig('field-1d-cascade.pdf', dpi='figure')


#  colors = plt.get_cmap('tab10').colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
_, axes = plt.subplots(2, 1, sharex=True,
                       figsize=(12*TO_INCH, 10*TO_INCH), dpi=300)
S = generate_strain(4, 10, 100, 200, 7)
axes[0].plot(t, S)
axes[0].set_ylabel(r'$w(t)$')
#  axes[1].plot(t, generate_strained_field(4, 10, .4, 100, 200, 7),
#               label=r'$\tau=0.4$')
axes[1].plot(t, generate_strained_field(4, 10, .3, 100, 200, 7),
             label=r'$\tau=0.3$', color=colors[3])
axes[1].plot(t, generate_strained_field(4, 10, .2, 100, 200, 7),
             label=r'$\tau=0.2$', color=colors[2])
axes[1].plot(t, generate_strained_field(4, 10, .1, 100, 200, 7),
             label=r'$\tau=0.1$', color=colors[1])
axes[1].plot(t, generate_strained_field(4, 10, .0, 100, 200, 7),
             label=r'$\tau=0$', color=colors[0])
axes[1].set_ylabel(r'$u(t)$')
axes[1].set_xlabel(r'$t$')
axes[1].set_xlim(0, 1)
axes[1].legend(loc='lower left', ncol=1)
plt.tight_layout(pad=.27)
plt.savefig('field-1d-strain.pdf', dpi='figure')

print(np.max(np.abs(generate_fBm_intermittent(4, 10, 100, 200, 7)\
                    -generate_strained_field(4, 10, 0, 100, 200, 7))))


_, axes = plt.subplots(2, 1, sharex=True,
                      figsize=(12*TO_INCH, 10*TO_INCH), dpi=300)
u = generate_strained_field(4, 12, .2, 100, 200, 10)
du = u[8:] - u[:-8]
axes[0].plot(np.linspace(0, 1, u.size), u)
axes[1].plot(np.linspace(0, 1, du.size), du)
axes[1].set_xlim(0, 1)
axes[1].set_xlabel(r'$t$')
axes[0].set_ylabel(r'$u(t)$')
axes[1].set_ylabel(r'$\delta u_8(t)$')
plt.tight_layout(pad=.27)
plt.savefig('field-1d-strain-increments.pdf', dpi='figure')




#  vim: set ff=unix tw=79 sw=4 ts=8 et ic ai :
