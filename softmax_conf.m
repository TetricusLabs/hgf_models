function [pvec, pstruct] = tapas_softmax_mu3_transp(r, ptrans)
  pvec    = [];
  pstruct = struct;
  return
  

function [logp, yhat, res] = tapas_softmax_mu3(r, infStates, ptrans)
  % Calculates the log-probability of responses under the softmax model
  %
  % --------------------------------------------------------------------------------------------------
  % Copyright (C) 2017-2019 Christoph Mathys, TNU, UZH & ETHZ
  %
  % This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
  % Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
  % (either version 3 or, at your option, any later version). For further details, see the file
  % COPYING or <http://www.gnu.org/licenses/>.
  
  % Predictions or posteriors?
  pop = 1; % Default: predictions
  if r.c_obs.predorpost == 2
      pop = 3; % Alternative: posteriors
  end
  
  % Initialize returned log-probabilities, predictions,
  % and residuals as NaNs so that NaN is returned for all
  % irregualar trials
  n = size(infStates,1);
  logp = NaN(n,1);
  yhat = NaN(n,1);
  res  = NaN(n,1);
  
  % Assumed structure of infStates:
  % dim 1: time (ie, input sequence number)
  % dim 2: HGF level
  % dim 3: choice number
  % dim 4: 1: muhat, 2: sahat, 3: mu, 4: sa
  
  % Number of choices
  nc = size(infStates,3);
  
  % Belief trajectories at 1st level
  states = squeeze(infStates(:,1,:,pop));
  
  % Log-volatility trajectory
  mu3 = squeeze(infStates(:,3,1,3));
  
  % Responses
  y = r.y(:,1);
  
  % Weed irregular trials out from inferred states and responses
  states(r.irr,:) = [];
  mu3(r.irr) = [];
  y(r.irr) = [];
  
  % Inverse decision temperature
  be = exp(-mu3);
  be = repmat(be,1,nc);
  
  % Partition functions
  Z = sum(exp(be.*states),2);
  Z = repmat(Z,1,nc);
  
  % Softmax probabilities
  prob = exp(be.*states)./Z;
  
  % Extract probabilities of chosen options
  probc = prob(sub2ind(size(prob), 1:length(y), y'));
  
  % Calculate log-probabilities for non-irregular trials
  reg = ~ismember(1:n,r.irr);
  logp(reg) = log(probc);
  yhat(reg) = probc;
  res(reg) = -log(probc);
  
  end
  

function c = tapas_softmax_mu3_config()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the softmax observation model for multinomial responses with phasic
% volatility exp(mu3) as the decision temperature
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2017 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Config structure
c = struct;

% Is the decision based on predictions or posteriors? Comment as appropriate.
c.predorpost = 1; % Predictions
%c.predorpost = 2; % Posteriors

% Model name
c.model = 'softmax_mu3';

% Sufficient statistics of Gaussian parameter priors

% Gather prior settings in vectors
c.priormus = [
         ];

c.priorsas = [
         ];

% Model filehandle
c.obs_fun = @tapas_softmax_mu3;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_softmax_mu3_transp;

return;
