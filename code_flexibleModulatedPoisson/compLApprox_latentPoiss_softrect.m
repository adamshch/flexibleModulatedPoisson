function [xmap,negH] = compLApprox_latentPoiss_softrect(r,mu,sig)

% [xmap,negH] = compLApprox_latentPoiss_softrect(r,mu,sig)
%
% Compute laplace approximation to posterior over latent noise variable
% under latent-gaussian Poisson model with soft-rect nonlinearity ('softrect')
%
% Inputs: 
%  r  -  spike count vector or matrix
%  mu - mean (scalar or size-r matrix)
%  sig - std deviation (scalar or size-r matrix)
%
% Outputs:
%  xmap - posterior maximum of the zero-mean noise variable
%  negH - neg Hessian of log-posterior at mode, which is (1/variance)
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make anonymous function for log-posterior
flogpost = @(r,f,xmap,sig)(r.*log(f)-f - xmap.^2./(2*sig.^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---- Initialize MAP estimate ----------
xmap = zeros(size(r));
ginv = @(y)(log(exp(y)-1));                                                % inverse function
% Initialize for r=0 responses
ii0 =(r==0);                                                               % Indices for r=0 responses
if any(ii0(:))
    % midpoint between where prior and likelihood cross 0.5
    xp5pri    = -sig(ii0)*sqrt(2*log(2));                                  % prior hits 0.5
    xp5li     = (log(exp(log(2))-1))-mu(ii0);                              % likelihood hits 0.5
    xmap(ii0) = (xp5pri+xp5li*4)/5;                                        % mean of those two
end
% Initialize for r>0 trials using Laplace approximation
ii1 = ~ii0;                                                                % Indices for r=1 responses
if any(ii1(:))
    limax      = ginv(r(ii1))-mu(ii1);                                     % argmax of likelihood
    [f,df,ddf] = softrect(limax+mu(ii1));                                  % Soft-rectification function evaluation
    negH       = -r(ii1).*(f.*ddf-df.^2)./f.^2 + ddf;                      % Hessian
    xmap(ii1)  = limax./(1./(sig(ii1).^2.*negH)+1);                        % Bayesian update
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---- Run finite number of Newton steps -----
iter       = 0;                                                            % Initialize iteration number 
maxiter    = 20;                                                           % Set maximum number of iterations
[f,df,ddf] = softrect(xmap+mu);                                            % Initial evaluation of the function
logpost    = flogpost(r,f,xmap,sig);                                       % Initial value of log-posterior
g          = -xmap./sig.^2 + r.*(df./f) - df;                              % Initial gradient
while any((abs(g(:))>1e-8)) && iter<maxiter
    negH = 1./sig.^2 -r.*(f.*ddf-df.^2)./f.^2 + ddf;                       % Calculate the Hessian
    
    % Determine step size
    xstp       = g./abs(negH);                                             % Calculate the Newton step size
    xstp       = sign(xstp).*min(abs(xstp),100);                           % heuristic limit max step size
    xnew       = xmap+xstp;                                                % new x location
    logpostnew = flogpost(r,softrect(xnew+mu),xnew,sig);                   % new value of log-posterior
    
    % loop to reduce step size, if necessary
    ii = (logpostnew<logpost) & (abs(xstp)>1e-5);                          % indices with a decrease
    while any(ii(:))
        xstp(ii)    = xstp(ii)/2;                                          % reduce step size by 2
        xnew(ii)    = xmap(ii)+xstp(ii);                                   % compute new value
        logpostnew(ii) = flogpost(r(ii),softrect(xnew(ii)+mu(ii)),...
                                                        xnew(ii),sig(ii)); % new value of log-post
        ii(ii)      = (logpostnew(ii)<logpost(ii)) & (abs(xstp(ii))>1e-5); % indices with a decrease
    end
    
    % Update MAP estimate
    xmap    = xnew;
    logpost = logpostnew;
    iter    = iter+1;
    
    % Compute new value of function for next iter
    if iter < maxiter
        [f,df,ddf] = softrect(xmap+mu);                                    % Query the new value of function
        g          = -xmap./sig.^2 + r.*(df./f) - df;                      % Calculate the new gradient
    end
end
if iter==maxiter
    warning('Newton-method may not have converged: check %s.m', mfilename); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
