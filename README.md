Bsc Thesis Scripts
---

* `alpert_rokhlin_wavelets.py`: analytic solution of the 1d wavelet integrals,
  wavelet coefficients for `dµ=dx` up to order `q=5`
* `mra_sympy.py`: solving the wavelet coefficients using `sympy`'s linear
  algebra capabilities
* `field_*d_implementation*.py`: implementation of the processes; the 3d case
  uses numerically precomputed wavelet integrals
* `field_*d_plots.py`: example plots of the processes and their wavelet
  integrals
* `field_*d_statistics.py`, `cascade_analysis.py`: code to generate samples and
  analyze the statistics of the final processes and the multiplicative cascade
* `field_3d_util_precomp.py`: utility functions to provide the precomputed
  wavelet integrals
* `wavelet_data`: the precomputed wavelet integrals (*no incompressibility!*)


>  vim: set ff=unix tw=79 sw=4 ts=4 et ic ai :
