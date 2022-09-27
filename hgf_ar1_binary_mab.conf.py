# --------------------------------------------------------------------------------------------------
# Copyright (C) 2013-2017 Christoph Mathys, TNU, UZH & ETHZ

# This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
# Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
# (either version 3 or, at your option, any later version). For further details, see the file
# COPYING or <http://www.gnu.org/licenses/>.

import numpy as np
import numpy.matlib


def tapas_hgf_ar1_binary_mab_config():


    c = {}
    c.model = 'hgf_ar1_binary_mab'
    c.n_levels = 3
    c.n_bandits = 3
    c.coupled = False
    c.irregular_intervals = False

    c.mu_0mu = np.array([np.NaN, 0, 1])
    c.mu_0sa = np.array([np.NaN, 1, 1])
    c.logsa_0mu = np.array([np.NaN, np.log(0.1), np.log(1)])
    c.logsa_0sa = np.array([np.NaN, 1, 1])
    c.logitphimu = np.array([np.NaN, tapas_logit(0.4, 1), tapas_logit(0.2, 1)])
    c.logitphisa = np.array([np.NaN, 1, 1])
    c.mmu = np.array([np.NaN, c.mu_0mu(2), c.mu_0mu(3)])
    c.msa = np.array([np.NaN, 0, 0])
    c.logkamu = np.array([np.log(1), np.log(0.6)])
    c.logkasa = np.array([0, 0.1])
    c.ommu = np.array([np.NaN, - 2, - 2])
    c.omsa = np.array([np.NaN, 1, 1])
    c.priormus = np.concatenate(
        [c.mu_0mu, c.logsa_0mu, c.logitphimu, c.mmu, c.logkamu, c.ommu])
    c.priorsas = np.concatenate(
        [c.mu_0sa, c.logsa_0sa, c.logitphisa, c.msa, c.logkasa, c.omsa])
    expectedLength = 5 * c.n_levels + (c.n_levels - 1)
    if len(np.array([c.priormus, c.priorsas])) != 2 * expectedLength:
        raise Exception('tapas:hgf:PriorDefNotMatchingLevels',
                        'Prior definition does not match number of levels.')

    c.prc_fun = tapas_hgf_ar1_binary_mab
    c.transp_prc_fun = tapas_hgf_ar1_binary_mab_transp
    return c


def tapas_hgf_ar1_binary_mab_transp(r=None, ptrans=None):


    pvec = np.full([1, len(ptrans)], np.nan)
    pstruct = {}
    l = r.c_prc.n_levels
    pvec[np.arange[1, l+1]] = ptrans(np.arange(1, l+1))

    pstruct.mu_0 = pvec(np.arange(1, l+1))
    pvec[np.arange[l + 1, 2 * l+1]] = np.exp(ptrans(np.arange(l + 1, 2 * l+1)))

    pstruct.sa_0 = pvec(np.arange(l + 1, 2 * l+1))
    pvec[np.arange[2 * l + 1, 3 * l+1]
         ] = tapas_sgm(ptrans(np.arange(2 * l + 1, 3 * l+1)), 1)

    pstruct.phi = pvec(np.arange(2 * l + 1, 3 * l+1))
    pvec[np.arange[3 * l + 1, 4 * l+1]] = ptrans(np.arange(3 * l + 1, 4 * l+1))

    pstruct.m = pvec(np.arange(3 * l + 1, 4 * l+1))
    pvec[np.arange[4 * l + 1, 5 * l - 1+1]
         ] = np.exp(ptrans(np.arange(4 * l + 1, 5 * l - 1+1)))

    pstruct.ka = pvec(np.arange(4 * l + 1, 5 * l - 1+1))
    pvec[np.arange[5 * l, 6 * l - 1+1]] = ptrans(np.arange(5 * l, 6 * l - 1+1))

    pstruct.om = pvec(np.arange(5 * l, 6 * l - 1+1))
    return pvec, pstruct


def tapas_hgf_ar1_binary_mab(r=None, p=None, varargin=None):
    if not len(varargin) == 0 and str(varargin[0]) == str('trans'):
        p = tapas_hgf_ar1_binary_mab_transp(r, p)

    l = r.c_prc.n_levels
    if l % 5:
      raise RuntimeError('Cannot determine number of levels')

    b = r.c_prc.n_bandits
    if not b:
      raise RuntimeError('Number of bandits must be configured')

    coupled = False
    if r.c_prc.coupled == True:
        if b == 2:
            coupled = True
        else:
            raise Exception('tapas:hgf:HgfBinaryMab:CoupledOnlyForTwo',
                            'Coupled updating can only be configured for 2 bandits.')

    mu_0 = p[0:l]
    sa_0 = p[l:2 * l]
    
    # the l seems to bounce from and back from 0-based indexing to 1-based indexing
    # i can infer the indexes from looking at some matlab -> python code but
    # can't find the indexing easily for phi or m or u
    phi = p[l+1:2*l]
    m = p[2*l+1:3*l]
    u = np.array([0, r.u[:, 1]])

    # these values should be okay
    ka = p[3*l:4*l-1]
    om = p[4*l-1:5*l-2]
    with np.errstate(divide='ignore'): th = np.exp(p[5*l-2])

    try:
        y = np.array([1, r.y[:, 1]])
        irr = r.irr
    finally:
        pass

    n = u.shape[0]
    if r.c_prc.irregular_intervals:
        if u.shape[2-1] > 1:
            t = np.array(0, r.u[:, -1])
        else:
            raise Exception('tapas:hgf:InputSingleColumn', 'Input matrix must contain more than one column if irregular_intervals is set to true.')
    else:
        t = np.ones((n, 1))


    mu = np.empty(n, l, b) * np.nan
    pi = np.empty(n, l, b) * np.nan
    muhat = np.empty(n, l, b) * np.nan
    pihat = np.empty(n, l, b) * np.nan
    v = np.empty(n, l) * np.nan
    w = np.empty(n, l - 1) * np.nan
    da = np.empty(n, l) * np.nan
    mu[0, 0, :] = tapas_sgm(mu_0[0], 1)
    muhat[0, 0, :] = mu[0, 0, :]

    # is 0 correct here? Or is it 0 indexed and should be -1?
    pihat[0, 0, :] = 0
    pi[0, 0, :] = np.Inf
    mu[0, 1:, :] = np.matlib.repmat(mu_0[1:], np.array([1, 1, b]))
    pi[1, np.arange[2, end()+1], :] = np.matlib.repmat(1.0 / sa_0(np.arange(2, end()+1)), np.array([1, 1, b]))
    for k in np.arange(2, n+1, 1).reshape(-1):
        if not ismember(k - 1, r.ign):
            muhat[k, 2, :] = mu(k - 1, 2, :) + t(k) * phi(2) * (m(2) - mu(k - 1, 2, : ))
            muhat[k, 1, :] = tapas_sgm(ka(1) * muhat(k, 2, :), 1)
            pihat[k, 1, :] = 1 / (np.multiply(muhat(k, 1, : ), (1 - muhat(k, 1, : ))))
            pi[k, 1, :] = pihat(k, 1, : )
            pi[k, 1, y[k]] = np.Inf
            mu[k, 1, :] = muhat(k, 1, : )
            mu[k, 1, y[k]] = u(k)
            da[k, 1] = mu(k, 1, y(k)) - muhat(k, 1, y(k))
            pihat[k, 2, :] = 1 / (1 / pi(k - 1, 2, :) + np.exp(ka(2) * mu(k - 1, 3, : ) + om(2)))
            pi[k, 2, :] = pihat(k, 2, :) + ka(1) ** 2 / pihat(k, 1, : )
            mu[k, 2, :] = muhat(k, 2, : )
            mu[k, 2, y[k]] = muhat(k, 2, y(k)) + ka(1) / \
                pi(k, 2, y(k)) * da(k, 1)
            da[k, 2] = (1 / pi(k, 2, y(k)) + (mu(k, 2, y(k)) -
                        muhat(k, 2, y(k))) ** 2) * pihat(k, 2, y(k)) - 1
            if l > 3:
                for j in np.arange(3, l - 1+1).reshape(-1):
                    muhat[k, j, :] = mu(k - 1, j, :) + t(k) * phi(j) * (m(j) - mu(k - 1, j))
                    pihat[k, j, :] = 1 / (1 / pi(k - 1, j, :) + t(k) * np.exp(ka(j) * mu(k - 1, j + 1, : ) + om(j)))
                    v[k, j - 1] = t(k) * np.exp(ka(j - 1) *
                                                mu(k - 1, j, y(k)) + om(j - 1))
                    w[k, j - 1] = v(k, j - 1) * pihat(k, j - 1, y(k))
                    pi[k, j, :] = pihat(k, j, : ) + 1 / 2 * ka(j - 1) ** 2 * w(k, j - 1) * (w(k, j - 1) + (2 * w(k, j - 1) - 1) * da(k, j - 1))
                    if pi(k, j, 1) <= 0:
                        raise Exception(
                            'tapas:hgf:NegPostPrec', 'Negative posterior precision. Parameters are in a region where model assumptions are violated.')
                    mu[k, j, :] = muhat(k, j, :) + 1 / 2 * 1 / pi(k, j) * ka(j - 1) * w(k, j - 1) * da(k, j - 1)
                    da[k, j] = (1 / pi(k, j, y(k)) + (mu(k, j, y(k)) -
                                muhat(k, j, y(k))) ** 2) * pihat(k, j, y(k)) - 1
            muhat[k, l, :] = mu(k - 1, l, :) + t(k) * phi(l) * (m(l) - mu(k - 1, l))
            pihat[k, l, :] = 1 / (1 / pi(k - 1, l, : ) + t(k) * th)
            v[k, l] = t(k) * th
            v[k, l - 1] = t(k) * np.exp(ka(l - 1) *
                                        mu(k - 1, l, y(k)) + om(l - 1))
            w[k, l - 1] = v(k, l - 1) * pihat(k, l - 1, y(k))
            pi[k, l, :] = pihat(k, l, : ) + 1 / 2 * ka(l - 1) ** 2 * w(k, l - 1) * (w(k, l - 1) + (2 * w(k, l - 1) - 1) * da(k, l - 1))
            if pi(k, l, 1) <= 0:
                raise Exception(
                    'tapas:hgf:NegPostPrec', 'Negative posterior precision. Parameters are in a region where model assumptions are violated.')
            mu[k, l, :] = muhat(k, l, :) + 1 / 2 * 1 / pi(k, l, : ) * ka(l - 1) * w(k, l - 1) * da(k, l - 1)
            da[k, l] = (1 / pi(k, l, y(k)) + (mu(k, l, y(k)) -
                        muhat(k, l, y(k))) ** 2) * pihat(k, l, y(k)) - 1
            if coupled == True:
                if y(k) == 1:
                    mu[k, 1, 2] = 1 - mu(k, 1, 1)
                    mu[k, 2, 2] = tapas_logit(1 - tapas_sgm(mu(k, 2, 1), 1), 1)
                else:
                    if y(k) == 2:
                        mu[k, 1, 1] = 1 - mu(k, 1, 2)
                        mu[k, 2, 1] = tapas_logit(
                            1 - tapas_sgm(mu(k, 2, 2), 1), 1)
        else:
            mu[k, :, :] = mu(k - 1, :, : )
            pi[k, :, :] = pi(k - 1, :, : )
            muhat[k, :, :] = muhat(k - 1, :, : )
            pihat[k, :, :] = pihat(k - 1, :, : )
            v[k, :] = v(k - 1, : )
            w[k, :] = w(k - 1, : )
            da[k, :] = da(k - 1, : )

    mu[1, :, :] = []
    pi[1, :, :] = []
    if np.any(np.isnan(mu)) or np.any(np.isnan(pi)):
        raise Exception('tapas:hgf:VarApproxInvalid',
                        'Variational approximation invalid. Parameters are in a region where model assumptions are violated.')
    else:
        dmu = np.diff(mu(: , np.arange(2, end()+1)))
        dpi = np.diff(pi(: , np.arange(2, end()+1)))
        rmdmu = np.matlib.repmat(np.sqrt(mean(dmu ** 2)), len(dmu), 1)
        rmdpi = np.matlib.repmat(np.sqrt(mean(dpi ** 2)), len(dpi), 1)
        jumpTol = 16
        if np.any(np.abs(dmu) > jumpTol * rmdmu) or np.any(np.abs(dpi) > jumpTol * rmdpi):
            raise Exception('tapas:hgf:VarApproxInvalid',
                            'Variational approximation invalid. Parameters are in a region where model assumptions are violated.')

    muhat[1, :, :] = []
    pihat[1, :, :] = []
    v[1, :] = []
    w[1, :] = []
    da[1, :] = []
    y[1] = []
    yreg = y
    yreg[irr] = []
    mu2 = np.squeeze(mu(:, 2, : ))
    mu2[irr, :] = []
    mu2obs = mu2(sub2ind(mu2.shape, np.transpose(
        (np.arange(1, mu2.shape[1-1]+1))), yreg))
    mu1hat = np.squeeze(muhat(:, 1, : ))
    mu1hat[irr, :] = []
    mu1hatobs = mu1hat(sub2ind(mu1hat.shape, np.transpose(
        (np.arange(1, mu1hat.shape[1-1]+1))), yreg))
    upd1 = tapas_sgm(ka(1) * mu2obs, 1) - mu1hatobs
    dareg = da
    dareg[irr, :] = []
    lr1reg = upd1 / dareg(:, 1)
    lr1 = np.full([n - 1, 1], np.nan)
    lr1[setdiff[np.arange[1, n - 1+1], irr]] = lr1reg
    traj = {}
    traj.mu = mu
    traj.sa = 1.0 / pi
    traj.muhat = muhat
    traj.sahat = 1.0 / pihat
    traj.v = v
    traj.w = w
    traj.da = da
    traj.ud = mu - muhat
    psi = np.full([n - 1, l], np.nan)
    pi2 = np.squeeze(pi(:, 2, : ))
    pi2[irr, :] = []
    pi2obs = pi2(sub2ind(pi2.shape, np.transpose(
        (np.arange(1, pi2.shape[1-1]+1))), yreg))
    psi[setdiff[np.arange[1, n - 1+1], irr], 2] = 1.0 / pi2obs
    for i in np.arange(3, l+1).reshape(-1):
        pihati = np.squeeze(pihat(:, i - 1, : ))
        pihati[irr, :] = []
        pihatiobs = pihati(sub2ind(pihati.shape, np.transpose(
            (np.arange(1, pihati.shape[1-1]+1))), yreg))
        pii = np.squeeze(pi(:, i, : ))
        pii[irr, :] = []
        piiobs = pii(sub2ind(pii.shape, np.transpose(
            (np.arange(1, pii.shape[1-1]+1))), yreg))
        psi[setdiff[np.arange[1, n - 1+1], irr], i] = pihatiobs / piiobs

    traj.psi = psi
    epsi = np.full([n - 1, l], np.nan)
    epsi[:, np.arange[2, l+1]] = np.multiply(psi(: , np.arange(2, l+1)), da(: , np.arange(1, l - 1+1)))
    traj.epsi = epsi
    wt = np.full([n - 1, l], np.nan)
    wt[:, 1] = lr1
    wt[:, 2] = psi(: , 2)
    wt[:, np.arange[3, l+1]] = np.multiply(1 / 2 * (v(:, np.arange(2, l - 1+1)) * diag(ka(np.arange(2, l - 1+1)))), psi(: , np.arange(3, l+1)))
    traj.wt = wt
    infStates = np.full([n - 1, l, b, 4], np.nan)
    infStates[:, :, :, 1] = traj.muhat
    infStates[:, :, :, 2] = traj.sahat
    infStates[:, :, :, 3] = traj.mu
    infStates[:, :, :, 4] = traj.sa
    return traj, infStates
