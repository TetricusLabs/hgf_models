# Copyright (C) 2017 Christoph Mathys, TNU, UZH & ETHZ

# This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
# Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
# (either version 3 or, at your option, any later version). For further details, see the file
# COPYING or <http://www.gnu.org/licenses/>.


import numpy as np
import numpy.matlib


def ismember(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = np.isin(a_vec, b_vec)
    common = a[bool_ind]
    common_unique, common_inv = np.unique(common, return_inverse=True)
    b_unique, b_ind = np.unique(b_vec, return_index=True)
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]


def tapas_softmax_mu3(r=None, infStates=None, ptrans=None):



    pop = 0

    if r['c_obs']['predorpost'] == 1:
        pop = 2

    n = infStates.shape[0]
    logp = np.full([n, 1], np.nan)
    yhat = np.full([n, 1], np.nan)
    res = np.full([n, 1], np.nan)

    nc = infStates.shape[2]
    states = np.squeeze(infStates[:, 0, :, pop])
    mu3 = np.squeeze(infStates[:, 2, 0, 2])
    y = r['y'][:, 0]
    # ????
    # states[r.irr, :] = np.array([])
    states = np.delete(states, [r['irr'], :])
    mu3 = np.delete(mu3, r['irr'])
    y = np.delete(y, r['irr'])
    be = np.exp(- mu3)
    be = np.matlib.repmat(be, 1, nc)
    Z = np.sum(np.exp(np.multiply(be, states)), 1)
    Z = np.matlib.repmat(Z, 1, nc)
    prob = np.exp(np.multiply(be, states)) / Z
    probc = prob(sub2ind(prob.shape, np.arange(1, len(y)+1), np.transpose(y)))
    reg = not ismember(np.arange(1, n+1), r['irr'])
    logp[reg] = np.log(probc)
    yhat[reg] = probc
    res[reg] = - np.log(probc)
    return logp, yhat, res


def tapas_softmax_mu3_config():    
    
    c = {}
    c.predorpost = 1
    c.model = 'softmax_mu3'

    c.priormus = np.array([])
    c.priorsas = np.array([])
    c.obs_fun = tapas_softmax_mu3
    c.transp_obs_fun = lambda r, ptrans: {
        'pvec': np.array([]),
        'pstruct': {}
    }
    return c
