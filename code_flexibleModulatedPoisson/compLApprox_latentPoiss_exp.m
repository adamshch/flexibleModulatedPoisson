function [xmap,negH] = compLApprox_latentPoiss_exp(r,mu,sig)
% [xmap,negH] = compLApprox_latentPoiss_exp(r,mu,sig)
%
% Compute laplace approximation to posterior over latent noise variable
% under latent-gaussian Poisson model with exponential nonlinearity.

% Inputs: 
%    r [m x n] - spike count vector or matrix
%   mu [m x n] - mean vector or matrix)
%  sig [m x n] - std deviation (scalar or size-r matrix)
%
% Outputs:
%  xmap - posterior maximum of the zero-mean noise variable
%  negH - neg Hessian of log-posterior at mode, which is (1/variance)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---  Initialize MAP estimate ---------
xmap = zeros(size(r));
ii0  = (r==0);                                                             % Initialize for r=0 trials

if any(ii0(:))
    % midpoint between where prior and likelihood cross 0.5
    xp5pri    = -sig(ii0)*sqrt(2*log(2));                                  % prior hits 0.5
    xp5li     = log(log(2))-mu(ii0);                                       % likelihood hits 0.5
    xmap(ii0) = (xp5pri+xp5li)/2;                                          % mean of those two
end
% Initialize for r>0 trials using Laplace approximation
ii1 = ~ii0;
if any(ii1(:))
    limax     = log(r(ii1))-mu(ii1);                                       % argmax of likelihood
    xmap(ii1) = limax./(1./(sig(ii1).^2.*r(ii1))+1);                       % Bayesian update
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---- Run finite number of Newton steps -----
iter    = 0;                                                               % Initialize the iterations
maxiter = 10;                                                              % Set the max number of iterations
g       = inf;                                                             % Initialize the gradient to infinity
while any((abs(g(:))>1e-8)) && iter<maxiter
    g    = -xmap./sig.^2 - exp(xmap+mu)+r;                                 % Calculate the gradient
    negH = exp(xmap+mu)+1./sig.^2;                                         % Calculate the negative Hessian

    % Update MAP estimate
    xmap = xmap+g./negH;                                                   % Take a Newton step
    iter = iter+1;                                                         % Increase the iteration number
end
if iter==maxiter
    warning('Newton-method may not have converged: check %s.m', mfilename); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
