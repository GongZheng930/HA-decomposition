'''Standard Incomplete Market model with Endogenous Labor Supply'''

import numpy as np
from numba import vectorize, njit
import sequence_jacobian as sj

from sequence_jacobian import het
from sequence_jacobian import interpolate


def hh_init(borr_cons, a_grid, we, r, eis, T):
    fininc =  a_grid + T[:, np.newaxis] - borr_cons / (1 + r)
    coh = a_grid[np.newaxis, :] + we[:, np.newaxis] + T[:, np.newaxis]
    Va = (0.1 * coh) ** (-1 / eis)
    return fininc, Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(borr_cons, Va_p, a_grid, we, T, r, beta, eis, frisch, vphi):
    '''Single backward step via EGM.'''
    uc_nextgrid = beta * (1 + r) * Va_p
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, np.newaxis], eis, frisch, vphi)

    lhs = c_nextgrid - we[:, np.newaxis] * n_nextgrid + a_grid[np.newaxis, :] / (1 + r) - T[:, np.newaxis]
    rhs = a_grid

    c = interpolate.interpolate_y(lhs, rhs, c_nextgrid)
    n = interpolate.interpolate_y(lhs, rhs, n_nextgrid)

    a = (rhs + we[:, np.newaxis] * n + T[:, np.newaxis] - c) * (1 + r)
    iconst = np.nonzero(a < borr_cons)
    a[iconst] = borr_cons

    if iconst[0].size != 0 and iconst[1].size != 0:
        c[iconst], n[iconst] = solve_cn(we[iconst[0]],
                                        rhs[iconst[1]] + T[iconst[0]] - borr_cons / (1 + r) ,
                                        eis, frisch, vphi, Va_p[iconst])

    Va = c ** (-1 / eis)
    
    return Va, a, c, n


'''Supporting functions for HA block'''

@njit
def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters"""
    return uc ** (-eis), (w * uc / vphi) ** frisch


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


@vectorize
def solve_uc(w, T, eis, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/eis) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """
    log_uc =  np.log(uc_seed) #uc_seed.apply(np.log)
    for i in range(30):
        ne, ne_p = netexp(log_uc, w, T, eis, frisch, vphi)
        if abs(ne) < 1E-11:
            break
        else:
            log_uc -= ne / ne_p
    else:
        raise ValueError("Cannot solve constrained household's problem: No convergence after 30 iterations!")

    return np.exp(log_uc)


@njit
def netexp(log_uc, w, T, eis, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative."""
    c, n = cn(np.exp(log_uc), w, eis, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -eis and frisch wrt log u'(c)
    c_loguc = -eis * c
    n_loguc = frisch * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc
