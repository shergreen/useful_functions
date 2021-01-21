"""
NFW Halo Subroutines

Default units are such that r_vir=M_vir=1

Sheridan B. Green
sheridan.green@yale.edu
Fall 2019

TODO: Modify so that you can switch between r_s=1 and r_vir=1 easily.
TODO: Make the functions consistent and not need both rvir and M
"""

import numpy as np
from scipy.integrate import quad


def f(c):
    return np.log(1. + c) - c/(1. + c)


def mass(r, c, rvir=1., M=1.):
    return M * f(r*c / rvir) / f(c)


def density(r, c, M=1.):
    rho_s = (4. * np.pi * c**-3 * f(c))**-1
    return rho_s / ((r * c) * (1. + r*c)**2)


def sigmar(r, c, beta=0., M=1.):
    val = (r**(-2. * beta) / density(r, c, M)) * quad(lambda x: density(x,
                                                                        c, M) * mass(x, c, M) * x**(2.*beta - 2), r, np.inf)[0]
    return np.sqrt(val)


def potential(r, c, rvir=1., M=1.):
    vvir = np.sqrt(M / rvir)
    return -vvir**2 * np.log(1. + c*r/rvir) / (f(c)*r/rvir)
