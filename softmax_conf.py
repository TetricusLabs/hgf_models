import numpy as np
import numpy.matlib
    
def tapas_softmax_mu3(r = None,infStates = None,ptrans = None): 
    # Calculates the log-probability of responses under the softmax model
    
    # --------------------------------------------------------------------------------------------------
    # Copyright (C) 2017-2019 Christoph Mathys, TNU, UZH & ETHZ
    
    # This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
    # Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
    # (either version 3 or, at your option, any later version). For further details, see the file
    # COPYING or <http://www.gnu.org/licenses/>.
    
    # Predictions or posteriors?
    pop = 1
    
    if r.c_obs.predorpost == 2:
        pop = 3
    
    # Initialize returned log-probabilities, predictions,
    # and residuals as NaNs so that NaN is returned for all
    # irregualar trials
    n = infStates.shape[1-1]
    logp = np.full([n,1],np.nan)
    yhat = np.full([n,1],np.nan)
    res = np.full([n,1],np.nan)
    # Assumed structure of infStates:
    # dim 1: time (ie, input sequence number)
    # dim 2: HGF level
    # dim 3: choice number
    # dim 4: 1: muhat, 2: sahat, 3: mu, 4: sa
    
    # Number of choices
    nc = infStates.shape[3-1]
    # Belief trajectories at 1st level
    states = np.squeeze(infStates[:,1,:,pop])
    # Log-volatility trajectory
    mu3 = np.squeeze(infStates[:,3,1,3])
    # Responses
    y = r.y[:,1]
    # Weed irregular trials out from inferred states and responses
    states[r.irr,:] = []
    mu3[r.irr] = []
    y[r.irr] = []
    # Inverse decision temperature
    be = np.exp(- mu3)
    be = np.matlib.repmat(be,1,nc)
    # Partition functions
    Z = np.sum(np.exp(np.multiply(be,states)), 2-1)
    Z = np.matlib.repmat(Z,1,nc)
    # Softmax probabilities
    prob = np.exp(np.multiply(be,states)) / Z
    # Extract probabilities of chosen options
    probc = prob(sub2ind(prob.shape,np.arange(1,len(y)+1),np.transpose(y)))
    # Calculate log-probabilities for non-irregular trials
    reg = not ismember(np.arange(1,n+1),r.irr) 
    logp[reg] = np.log(probc)
    yhat[reg] = probc
    res[reg] = - np.log(probc)
    return logp,yhat,res
    
    return logp,yhat,res


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
    
    #c.predorpost = 2; # Posteriors
    
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
    c.transp_obs_fun = lambda r,ptrans: {
      'pvec': [],
      'pstruct': {}
    }
    return c