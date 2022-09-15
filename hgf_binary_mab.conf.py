

import numpy as np
import numpy.matlib
    
def tapas_hgf_ar1_binary_mab_config(): 
    ####################################################################################################
    
    # Contains the configuration for the Hierarchical Gaussian Filter (HGF) for AR(1) processes in a
    # multi-armded bandit situation for binary inputs in the absence of perceptual uncertainty.
    
    # The HGF is the model introduced in
    
    # Mathys C, Daunizeau J, Friston, KJ, and Stephan KE. (2011). A Bayesian foundation
    # for individual learning under uncertainty. Frontiers in Human Neuroscience, 5:39.
    
    # The binary HGF model has since been augmented with a positive factor kappa1 which
    # scales the second level with respect to the first, i.e., the relation between the
    # first and second level is
    
    # p(x1=1|x2) = s(kappa1*x2), where s(.) is the logistic sigmoid.
    
    # By default, kappa1 is fixed to 1, leading exactly to the model introduced in
    # Mathys et al. (2011).
    
    # This file refers to BINARY inputs (Eqs 1-3 in Mathys et al., (2011));
    # for continuous inputs, refer to tapas_hgf_config.
    
    ####################################################################################################
    
    # The HGF configuration consists of the priors of parameters and initial values. All priors are
    # Gaussian in the space where the quantity they refer to is estimated. They are specified by their
    # sufficient statistics: mean and variance (NOT standard deviation).
    
    # Quantities are estimated in their native space if they are unbounded (e.g., the omegas). They are
    # estimated in log-space if they have a natural lower bound at zero (e.g., the sigmas).
    
    # The phis are estimated in 'logit space' because they are confined to the interval from 0 to 1.
    # 'Logit-space' is a logistic sigmoid transformation of native space with a variable upper bound
    # a>0:
    
    # tapas_logit(x) = ln(x/(a-x)); x = a/(1+exp(-tapas_logit(x)))
    
    # Parameters can be fixed (i.e., set to a fixed value) by setting the variance of their prior to
    # zero. Aside from being useful for model comparison, the need for this arises whenever the scale
    # and origin at the j-th level are arbitrary. This is the case if the observation model does not
    # contain the representations mu_j and sigma_j. A choice of scale and origin is then implied by
    # fixing the initial value mu_j_0 of mu_j and either kappa_j-1 or omega_j-1.
    
    # Fitted trajectories can be plotted by using the command
    
    # >> tapas_hgf_binary_mab_plotTraj(est)
    
    # where est is the stucture returned by tapas_fitModel. This structure contains the estimated
    # perceptual parameters in est.p_prc and the estimated trajectories of the agent's
    # representations (cf. Mathys et al., 2011). Their meanings are:
    
    #         est.p_prc.mu_0       row vector of initial values of mu (in ascending order of levels)
    #         est.p_prc.sa_0       row vector of initial values of sigma (in ascending order of levels)
    #         est.p_prc.phi        row vector of phis (representing reversion slope to attractor; in ascending order of levels)
    #         est.p_prc.m          row vector of ms (representing attractors; in ascending order of levels)
    #         est.p_prc.ka         row vector of kappas (in ascending order of levels)
    #         est.p_prc.om         row vector of omegas (in ascending order of levels)
    
    # Note that the first entry in all of the row vectors will be NaN because, at the first level,
    # these parameters are either determined by the second level (mu_0 and sa_0) or undefined (rho,
    # kappa, and omega).
    
    #         est.traj.mu          mu (rows: trials, columns: levels, 3rd dim: bandits)
    #         est.traj.sa          sigma (rows: trials, columns: levels, 3rd dim: bandits)
    #         est.traj.muhat       prediction of mu (rows: trials, columns: levels, 3rd dim: bandits)
    #         est.traj.sahat       precisions of predictions (rows: trials, columns: levels, 3rd dim: bandits)
    #         est.traj.v           inferred variance of random walk (rows: trials, columns: levels)
    #         est.traj.w           weighting factors (rows: trials, columns: levels)
    #         est.traj.da          volatility prediction errors  (rows: trials, columns: levels)
    #         est.traj.ud          updates with respect to prediction  (rows: trials, columns: levels)
    #         est.traj.psi         precision weights on prediction errors  (rows: trials, columns: levels)
    #         est.traj.epsi        precision-weighted prediction errors  (rows: trials, columns: levels)
    #         est.traj.wt          full weights on prediction errors (at the first level,
    #                                  this is the learning rate) (rows: trials, columns: levels)
    
    # Note that in the absence of sensory uncertainty (which is the assumption here), the first
    # column of mu, corresponding to the first level, will be equal to the inputs. Likewise, the
    # first column of sa will be 0 always.
    
    # Tips:
    # - When analyzing a new dataset, take your inputs u and responses y and use
    
    #   >> est = tapas_fitModel(y, u, 'tapas_hgf_ar1_binary_mab_config', 'tapas_bayes_optimal_binary_config');
    
    #   to determine the Bayes optimal perceptual parameters (given your current priors as defined in
    #   this file here, so choose them wide and loose to let the inputs influence the result). You can
    #   then use the optimal parameters as your new prior means for the perceptual parameters.
    
    # - If you get an error saying that the prior means are in a region where model assumptions are
    #   violated, lower the prior means of the omegas, starting with the highest level and proceeding
    #   downwards.
    
    # - Alternatives are lowering the prior means of the kappas, if they are not fixed, or adjusting
    #   the values of the kappas or omegas, if any of them are fixed.
    
    # - If the log-model evidence cannot be calculated because the Hessian poses problems, look at
    #   est.optim.H and fix the parameters that lead to NaNs.
    
    # - Your guide to all these adjustments is the log-model evidence (LME). Whenever the LME increases
    #   by at least 3 across datasets, the adjustment was a good idea and can be justified by just this:
    #   the LME increased, so you had a better model.
    
    # --------------------------------------------------------------------------------------------------
    # Copyright (C) 2013-2017 Christoph Mathys, TNU, UZH & ETHZ
    
    # This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
    # Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
    # (either version 3 or, at your option, any later version). For further details, see the file
    # COPYING or <http://www.gnu.org/licenses/>.
    
    # Config structure
    c = {}
    # Model name
    c.model = 'hgf_ar1_binary_mab'
    # Number of levels (minimum: 3)
    c.n_levels = 3
    # Number of bandits
    c.n_bandits = 3
    # Coupling
    # This may only be set to true if c.n_bandits is set to 2 above. If
    # true, it means that the two bandits' winning probabilities are
    # coupled in the sense that they add to 1 and are both updated on
    # each trial even though only the outcome for one of them is observed.
    c.coupled = False
    # Input intervals
    # If input intervals are irregular, the last column of the input
    # matrix u has to contain the interval between inputs k-1 and k
    # in the k-th row, and this flag has to be set to true
    c.irregular_intervals = False
    # Sufficient statistics of Gaussian parameter priors
    
    # Initial mus and sigmas
    # Format: row vectors of length n_levels
    # For all but the first two levels, this is usually best
    # kept fixed to 1 (determines origin on x_i-scale). The
    # first level is NaN because it is determined by the second,
    # and the second implies neutrality between outcomes when it
    # is centered at 0.
    c.mu_0mu = np.array([np.NaN,0,1])
    c.mu_0sa = np.array([np.NaN,1,1])
    c.logsa_0mu = np.array([np.NaN,np.log(0.1),np.log(1)])
    c.logsa_0sa = np.array([np.NaN,1,1])
    # Phis
    # Format: row vector of length n_levels.
    # Undefined (therefore NaN) at the first level.
    # Fix this to zero (-Inf in logit space) to set to zero.
    c.logitphimu = np.array([np.NaN,tapas_logit(0.4,1),tapas_logit(0.2,1)])
    c.logitphisa = np.array([np.NaN,1,1])
    # ms
    # Format: row vector of length n_levels.
    # This should be fixed for all levels where the omega of
    # the next lowest level is not fixed because that offers
    # an alternative parametrization of the same model.
    c.mmu = np.array([np.NaN,c.mu_0mu(2),c.mu_0mu(3)])
    c.msa = np.array([np.NaN,0,0])
    # Kappas
    # Format: row vector of length n_levels-1.
    # Fixing log(kappa1) to log(1) leads to the original HGF model.
    # Higher log(kappas) should be fixed (preferably to log(1)) if the
    # observation model does not use mu_i+1 (kappa then determines the
    # scaling of x_i+1).
    c.logkamu = np.array([np.log(1),np.log(0.6)])
    c.logkasa = np.array([0,0.1])
    # Omegas
    # Format: row vector of length n_levels.
    # Undefined (therefore NaN) at the first level.
    c.ommu = np.array([np.NaN,- 2,- 2])
    c.omsa = np.array([np.NaN,1,1])
    # Gather prior settings in vectors
    c.priormus = np.array([c.mu_0mu,c.logsa_0mu,c.logitphimu,c.mmu,c.logkamu,c.ommu])
    c.priorsas = np.array([c.mu_0sa,c.logsa_0sa,c.logitphisa,c.msa,c.logkasa,c.omsa])
    # Check whether we have the right number of priors
    expectedLength = 5 * c.n_levels + (c.n_levels - 1)
    if len(np.array([c.priormus,c.priorsas])) != 2 * expectedLength:
        raise Exception('tapas:hgf:PriorDefNotMatchingLevels','Prior definition does not match number of levels.')
    
    # Model function handle
    c.prc_fun = tapas_hgf_ar1_binary_mab
    # Handle to function that transforms perceptual parameters to their native space
    # from the space they are estimated in
    c.transp_prc_fun = tapas_hgf_ar1_binary_mab_transp
    return c
    
def tapas_hgf_ar1_binary_mab_transp(r = None,ptrans = None): 
    # --------------------------------------------------------------------------------------------------
            # Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
    
    # This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
            # Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
            # (either version 3 or, at your option, any later version). For further details, see the file
            # COPYING or <http://www.gnu.org/licenses/>.
    
    pvec = np.full([1,len(ptrans)],np.nan)
    pstruct = {}
    l = r.c_prc.n_levels
    pvec[np.arange[1,l+1]] = ptrans(np.arange(1,l+1))
    
    pstruct.mu_0 = pvec(np.arange(1,l+1))
    pvec[np.arange[l + 1,2 * l+1]] = np.exp(ptrans(np.arange(l + 1,2 * l+1)))
    
    pstruct.sa_0 = pvec(np.arange(l + 1,2 * l+1))
    pvec[np.arange[2 * l + 1,3 * l+1]] = tapas_sgm(ptrans(np.arange(2 * l + 1,3 * l+1)),1)
    
    pstruct.phi = pvec(np.arange(2 * l + 1,3 * l+1))
    pvec[np.arange[3 * l + 1,4 * l+1]] = ptrans(np.arange(3 * l + 1,4 * l+1))
    
    pstruct.m = pvec(np.arange(3 * l + 1,4 * l+1))
    pvec[np.arange[4 * l + 1,5 * l - 1+1]] = np.exp(ptrans(np.arange(4 * l + 1,5 * l - 1+1)))
    
    pstruct.ka = pvec(np.arange(4 * l + 1,5 * l - 1+1))
    pvec[np.arange[5 * l,6 * l - 1+1]] = ptrans(np.arange(5 * l,6 * l - 1+1))
    
    pstruct.om = pvec(np.arange(5 * l,6 * l - 1+1))
    return pvec,pstruct
    
def tapas_hgf_ar1_binary_mab(r = None,p = None,varargin = None): 
    # Calculates the trajectories of the agent's representations under the HGF in a multi-armed bandit
    # situation with binary outcomes
    
    # This function can be called in two ways:
    
    # (1) tapas_hgf_ar1_binary_mab(r, p)
    
    #     where r is the structure generated by tapas_fitModel and p is the parameter vector in native space;
    
    # (2) tapas_hgf_ar1_binary_mab(r, ptrans, 'trans')
    
    #     where r is the structure generated by tapas_fitModel, ptrans is the parameter vector in
    #     transformed space, and 'trans' is a flag indicating this.
    
    # --------------------------------------------------------------------------------------------------
    # Copyright (C) 2017 Christoph Mathys, TNU, UZH & ETHZ
    
    # This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
    # Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
    # (either version 3 or, at your option, any later version). For further details, see the file
    # COPYING or <http://www.gnu.org/licenses/>.
    
    # Transform paramaters back to their native space if needed
    if not len(varargin)==0  and str(varargin[0]) == str('trans'):
        p = tapas_hgf_ar1_binary_mab_transp(r,p)
    
    # Number of levels
    try:
        l = r.c_prc.n_levels
    finally:
        pass
    
    # Number of bandits
    try:
        b = r.c_prc.n_bandits
    finally:
        pass
    
    # Coupled updating
    # This is only allowed if there are 2 bandits. We here assume that the mu1hat for the two bandits
    # add to unity.
    coupled = False
    if r.c_prc.coupled == True:
        if b == 2:
            coupled = True
        else:
            raise Exception('tapas:hgf:HgfBinaryMab:CoupledOnlyForTwo','Coupled updating can only be configured for 2 bandits.')
    
    # Unpack parameters
    mu_0 = p(np.arange(1,l+1))
    sa_0 = p(np.arange(l + 1,2 * l+1))
    phi = p(np.arange(2 * l + 1,3 * l+1))
    m = p(np.arange(3 * l + 1,4 * l+1))
    ka = p(np.arange(4 * l + 1,5 * l - 1+1))
    om = p(np.arange(5 * l,6 * l - 2+1))
    th = np.exp(p(6 * l - 1))
    # Add dummy zeroth trial
    u = np.array([[0],[r.u(:,1)]])
    try:
        y = np.array([[1],[r.y(:,1)]])
        irr = r.irr
    finally:
        pass
    
    # Number of trials (including prior)
    n = u.shape[1-1]
    # Construct time axis
    if r.c_prc.irregular_intervals:
        if u.shape[2-1] > 1:
            t = np.array([[0],[r.u(:,end())]])
        else:
            raise Exception('tapas:hgf:InputSingleColumn','Input matrix must contain more than one column if irregular_intervals is set to true.')
    else:
        t = np.ones((n,1))
    
    # Initialize updated quantities
    
    # Representations
    mu = np.full([n,l,b],np.nan)
    np.pi = np.full([n,l,b],np.nan)
    # Other quantities
    muhat = np.full([n,l,b],np.nan)
    pihat = np.full([n,l,b],np.nan)
    v = np.full([n,l],np.nan)
    w = np.full([n,l - 1],np.nan)
    da = np.full([n,l],np.nan)
    # Representation priors
    # Note: first entries of the other quantities remain
    # NaN because they are undefined and are thrown away
    # at the end; their presence simply leads to consistent
    # trial indices.
    mu[1,1,:] = tapas_sgm(mu_0(2),1)
    muhat[1,1,:] = mu(1,1,:)
    pihat[1,1,:] = 0
    np.pi[1,1,:] = np.Inf
    mu[1,np.arange[2,end()+1],:] = np.matlib.repmat(mu_0(np.arange(2,end()+1)),np.array([1,1,b]))
    np.pi[1,np.arange[2,end()+1],:] = np.matlib.repmat(1.0 / sa_0(np.arange(2,end()+1)),np.array([1,1,b]))
    # Pass through representation update loop
    for k in np.arange(2,n+1,1).reshape(-1):
        if not_(ismember(k - 1,r.ign)):
            ######################
            # Effect of input u(k)
            ######################
            # 2nd level prediction
            muhat[k,2,:] = mu(k - 1,2,:) + t(k) * phi(2) * (m(2) - mu(k - 1,2,:))
            # 1st level
            # ~~~~~~~~~
            # Prediction
            muhat[k,1,:] = tapas_sgm(ka(1) * muhat(k,2,:),1)
            # Precision of prediction
            pihat[k,1,:] = 1 / (np.multiply(muhat(k,1,:),(1 - muhat(k,1,:))))
            # Updates
            np.pi[k,1,:] = pihat(k,1,:)
            np.pi[k,1,y[k]] = np.Inf
            mu[k,1,:] = muhat(k,1,:)
            mu[k,1,y[k]] = u(k)
            # Prediction error
            da[k,1] = mu(k,1,y(k)) - muhat(k,1,y(k))
            # 2nd level
            # ~~~~~~~~~
            # Prediction: see above
            # Precision of prediction
            pihat[k,2,:] = 1 / (1 / np.pi(k - 1,2,:) + np.exp(ka(2) * mu(k - 1,3,:) + om(2)))
            # Updates
            np.pi[k,2,:] = pihat(k,2,:) + ka(1) ** 2 / pihat(k,1,:)
            mu[k,2,:] = muhat(k,2,:)
            mu[k,2,y[k]] = muhat(k,2,y(k)) + ka(1) / np.pi(k,2,y(k)) * da(k,1)
            # Volatility prediction error
            da[k,2] = (1 / np.pi(k,2,y(k)) + (mu(k,2,y(k)) - muhat(k,2,y(k))) ** 2) * pihat(k,2,y(k)) - 1
            if l > 3:
                # Pass through higher levels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~
                for j in np.arange(3,l - 1+1).reshape(-1):
                    # Prediction
                    muhat[k,j,:] = mu(k - 1,j,:) + t(k) * phi(j) * (m(j) - mu(k - 1,j))
                    # Precision of prediction
                    pihat[k,j,:] = 1 / (1 / np.pi(k - 1,j,:) + t(k) * np.exp(ka(j) * mu(k - 1,j + 1,:) + om(j)))
                    # Weighting factor
                    v[k,j - 1] = t(k) * np.exp(ka(j - 1) * mu(k - 1,j,y(k)) + om(j - 1))
                    w[k,j - 1] = v(k,j - 1) * pihat(k,j - 1,y(k))
                    # Updates
                    np.pi[k,j,:] = pihat(k,j,:) + 1 / 2 * ka(j - 1) ** 2 * w(k,j - 1) * (w(k,j - 1) + (2 * w(k,j - 1) - 1) * da(k,j - 1))
                    if np.pi(k,j,1) <= 0:
                        raise Exception('tapas:hgf:NegPostPrec','Negative posterior precision. Parameters are in a region where model assumptions are violated.')
                    mu[k,j,:] = muhat(k,j,:) + 1 / 2 * 1 / np.pi(k,j) * ka(j - 1) * w(k,j - 1) * da(k,j - 1)
                    # Volatility prediction error
                    da[k,j] = (1 / np.pi(k,j,y(k)) + (mu(k,j,y(k)) - muhat(k,j,y(k))) ** 2) * pihat(k,j,y(k)) - 1
            # Last level
            # ~~~~~~~~~~
            # Prediction
            muhat[k,l,:] = mu(k - 1,l,:) + t(k) * phi(l) * (m(l) - mu(k - 1,l))
            # Precision of prediction
            pihat[k,l,:] = 1 / (1 / np.pi(k - 1,l,:) + t(k) * th)
            # Weighting factor
            v[k,l] = t(k) * th
            v[k,l - 1] = t(k) * np.exp(ka(l - 1) * mu(k - 1,l,y(k)) + om(l - 1))
            w[k,l - 1] = v(k,l - 1) * pihat(k,l - 1,y(k))
            # Updates
            np.pi[k,l,:] = pihat(k,l,:) + 1 / 2 * ka(l - 1) ** 2 * w(k,l - 1) * (w(k,l - 1) + (2 * w(k,l - 1) - 1) * da(k,l - 1))
            if np.pi(k,l,1) <= 0:
                raise Exception('tapas:hgf:NegPostPrec','Negative posterior precision. Parameters are in a region where model assumptions are violated.')
            mu[k,l,:] = muhat(k,l,:) + 1 / 2 * 1 / np.pi(k,l,:) * ka(l - 1) * w(k,l - 1) * da(k,l - 1)
            # Volatility prediction error
            da[k,l] = (1 / np.pi(k,l,y(k)) + (mu(k,l,y(k)) - muhat(k,l,y(k))) ** 2) * pihat(k,l,y(k)) - 1
            if coupled == True:
                if y(k) == 1:
                    mu[k,1,2] = 1 - mu(k,1,1)
                    mu[k,2,2] = tapas_logit(1 - tapas_sgm(mu(k,2,1),1),1)
                else:
                    if y(k) == 2:
                        mu[k,1,1] = 1 - mu(k,1,2)
                        mu[k,2,1] = tapas_logit(1 - tapas_sgm(mu(k,2,2),1),1)
        else:
            mu[k,:,:] = mu(k - 1,:,:)
            np.pi[k,:,:] = np.pi(k - 1,:,:)
            muhat[k,:,:] = muhat(k - 1,:,:)
            pihat[k,:,:] = pihat(k - 1,:,:)
            v[k,:] = v(k - 1,:)
            w[k,:] = w(k - 1,:)
            da[k,:] = da(k - 1,:)
    
    # Remove representation priors
    mu[1,:,:] = []
    np.pi[1,:,:] = []
    # Check validity of trajectories
    if np.any(np.isnan(mu)) or np.any(np.isnan(np.pi)):
        raise Exception('tapas:hgf:VarApproxInvalid','Variational approximation invalid. Parameters are in a region where model assumptions are violated.')
    else:
        # Check for implausible jumps in trajectories
        dmu = np.diff(mu(:,np.arange(2,end()+1)))
        dpi = np.diff(np.pi(:,np.arange(2,end()+1)))
        rmdmu = np.matlib.repmat(np.sqrt(mean(dmu ** 2)),len(dmu),1)
        rmdpi = np.matlib.repmat(np.sqrt(mean(dpi ** 2)),len(dpi),1)
        jumpTol = 16
        if np.any(np.abs(dmu) > jumpTol * rmdmu) or np.any(np.abs(dpi) > jumpTol * rmdpi):
            raise Exception('tapas:hgf:VarApproxInvalid','Variational approximation invalid. Parameters are in a region where model assumptions are violated.')
    
    # Remove other dummy initial values
    muhat[1,:,:] = []
    pihat[1,:,:] = []
    v[1,:] = []
    w[1,:] = []
    da[1,:] = []
    y[1] = []
    # Responses on regular trials
    yreg = y
    yreg[irr] = []
    # Implied learning rate at the first level
    mu2 = np.squeeze(mu(:,2,:))
    mu2[irr,:] = []
    mu2obs = mu2(sub2ind(mu2.shape,np.transpose((np.arange(1,mu2.shape[1-1]+1))),yreg))
    mu1hat = np.squeeze(muhat(:,1,:))
    mu1hat[irr,:] = []
    mu1hatobs = mu1hat(sub2ind(mu1hat.shape,np.transpose((np.arange(1,mu1hat.shape[1-1]+1))),yreg))
    upd1 = tapas_sgm(ka(1) * mu2obs,1) - mu1hatobs
    dareg = da
    dareg[irr,:] = []
    lr1reg = upd1 / dareg(:,1)
    lr1 = np.full([n - 1,1],np.nan)
    lr1[setdiff[np.arange[1,n - 1+1],irr]] = lr1reg
    # Create result data structure
    traj = struct
    traj.mu = mu
    traj.sa = 1.0 / np.pi
    traj.muhat = muhat
    traj.sahat = 1.0 / pihat
    traj.v = v
    traj.w = w
    traj.da = da
    # Updates with respect to prediction
    traj.ud = mu - muhat
    # Psi (precision weights on prediction errors)
    psi = np.full([n - 1,l],np.nan)
    pi2 = np.squeeze(np.pi(:,2,:))
    pi2[irr,:] = []
    pi2obs = pi2(sub2ind(pi2.shape,np.transpose((np.arange(1,pi2.shape[1-1]+1))),yreg))
    psi[setdiff[np.arange[1,n - 1+1],irr],2] = 1.0 / pi2obs
    for i in np.arange(3,l+1).reshape(-1):
        pihati = np.squeeze(pihat(:,i - 1,:))
        pihati[irr,:] = []
        pihatiobs = pihati(sub2ind(pihati.shape,np.transpose((np.arange(1,pihati.shape[1-1]+1))),yreg))
        pii = np.squeeze(np.pi(:,i,:))
        pii[irr,:] = []
        piiobs = pii(sub2ind(pii.shape,np.transpose((np.arange(1,pii.shape[1-1]+1))),yreg))
        psi[setdiff[np.arange[1,n - 1+1],irr],i] = pihatiobs / piiobs
    
    traj.psi = psi
    # Epsilons (precision-weighted prediction errors)
    epsi = np.full([n - 1,l],np.nan)
    epsi[:,np.arange[2,l+1]] = np.multiply(psi(:,np.arange(2,l+1)),da(:,np.arange(1,l - 1+1)))
    traj.epsi = epsi
    # Full learning rate (full weights on prediction errors)
    wt = np.full([n - 1,l],np.nan)
    wt[:,1] = lr1
    wt[:,2] = psi(:,2)
    wt[:,np.arange[3,l+1]] = np.multiply(1 / 2 * (v(:,np.arange(2,l - 1+1)) * diag(ka(np.arange(2,l - 1+1)))),psi(:,np.arange(3,l+1)))
    traj.wt = wt
    # Create matrices for use by the observation model
    np.infStates = np.full([n - 1,l,b,4],np.nan)
    np.infStates[:,:,:,1] = traj.muhat
    np.infStates[:,:,:,2] = traj.sahat
    np.infStates[:,:,:,3] = traj.mu
    np.infStates[:,:,:,4] = traj.sa
    return traj,infStates