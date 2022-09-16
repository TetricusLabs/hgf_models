import numpy as np
import numpy.matlib


def ismember(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = np.isin(a_vec, b_vec)
    common = a[bool_ind]
    # common = common_unique[common_inv]
    common_unique, common_inv = np.unique(common, return_inverse=True)
    # b_unique = b_vec[b_ind]
    b_unique, b_ind = np.unique(b_vec, return_index=True)
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]


def tapas_softmax_mu3(r=None, infStates=None, ptrans=None):
    # Calculates the log-probability of responses under the softmax model

    # --------------------------------------------------------------------------------------------------
    # Copyright (C) 2017-2019 Christoph Mathys, TNU, UZH & ETHZ

    # This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
    # Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
    # (either version 3 or, at your option, any later version). For further details, see the file
    # COPYING or <http://www.gnu.org/licenses/>.

    # Predictions or posteriors?
    # TODO: These should be 0 indexed?
    pop = 0

    if r.c_obs.predorpost == 1:
        pop = 2

    # Initialize returned log-probabilities, predictions,
    # and residuals as NaNs so that NaN is returned for all
    # irregualar trials
    n = infStates.shape[0]
    logp = np.full([n, 1], np.nan)
    yhat = np.full([n, 1], np.nan)
    res = np.full([n, 1], np.nan)
    # Assumed structure of infStates:
    # dim 1: time (ie, input sequence number)
    # dim 2: HGF level
    # dim 3: choice number
    # dim 4: 1: muhat, 2: sahat, 3: mu, 4: sa

    # Number of choices
    nc = infStates.shape[2]
    # Belief trajectories at 1st level
    states = np.squeeze(infStates[:, 0, :, pop])
    # Log-volatility trajectory
    mu3 = np.squeeze(infStates[:, 2, 0, 2])
    # Responses
    y = r.y[:, 0]
    # Weed irregular trials out from inferred states and responses
    states[r.irr, :] = []
    mu3[r.irr] = []
    y[r.irr] = []
    # Inverse decision temperature
    be = np.exp(- mu3)
    be = np.matlib.repmat(be, 1, nc)
    # Partition functions
    Z = np.sum(np.exp(np.multiply(be, states)), 1)
    Z = np.matlib.repmat(Z, 1, nc)
    # Softmax probabilities
    prob = np.exp(np.multiply(be, states)) / Z
    # Extract probabilities of chosen options
    # TODO: Should these be 0 indexed? np.arrange(0, len(y))?
    probc = prob(sub2ind(prob.shape, np.arange(1, len(y)+1), np.transpose(y)))
    # Calculate log-probabilities for non-irregular trials
    # TODO: Should these be 0 indexed? np.arrange(0, len(y))?
    reg = not ismember(np.arange(1, n+1), r.irr)
    logp[reg] = np.log(probc)
    yhat[reg] = probc
    res[reg] = - np.log(probc)
    return logp, yhat, res


def tapas_softmax_mu3_config():
    ####################################################################################################

    # Contains the configuration for the softmax observation model for multinomial responses with phasic
    # volatility exp(mu3) as the decision temperature

    ####################################################################################################

    # --------------------------------------------------------------------------------------------------
    # Copyright (C) 2017 Christoph Mathys, TNU, UZH & ETHZ

    # This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
    # Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
    # (either version 3 or, at your option, any later version). For further details, see the file
    # COPYING or <http://www.gnu.org/licenses/>.

    # Config structure
    c = {}
    # Is the decision based on predictions or posteriors? Comment as appropriate.
    c.predorpost = 1

    # c.predorpost = 2; # Posteriors

    # Model name
    c.model = 'softmax_mu3'
    # Sufficient statistics of Gaussian parameter priors

    # Gather prior settings in vectors
    c.priormus = []
    c.priorsas = []
    # Model filehandle
    c.obs_fun = tapas_softmax_mu3
    # Handle to function that transforms observation parameters to their native space
    # from the space they are estimated in
    c.transp_obs_fun = lambda r, ptrans: {
        'pvec': [],
        'pstruct': {}
    }
    return c
