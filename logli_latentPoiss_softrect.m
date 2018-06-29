function logL = logli_latentPoiss_softrect(r,mu,sig)

% logL = logli_latentPoiss_softrect(r,mu,sig)
%
% Compute loglikelihoods under latent-gaussian Poisson model with
% soft-rectifying nonlinearity ('softrect');
%
% Inputs: 
%  r  -  spike count vector or matrix
%  mu - mean (scalar or size-r matrix)
%  sig - std deviation (scalar or size-r matrix)
%
% Output: 
%  logL - log P(r | mu, sig) = \int P(r,x|mu,sig) dx
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make mu and sigma the same size as mu
rsz = size(r);
if length(mu)==1, mu = repmat(mu,rsz); end
if length(sig)==1, sig = repmat(sig,rsz); end

% Compute Laplace approximation to posterior
[xmap,negH] = compLApprox_latentPoiss_softrect(r,mu,sig);

% Compute integral using laplace approximation
f    = softrect(mu+xmap);                                                  % Pre-calculate the soft-rectification function
logL = -log(sig)-xmap.^2./(2*sig.^2) + ...                                 % prior Gaussian term
                             + r.*log(f) - f -gammaln(r+1) + ...           % Poisson likelihood term
                                                        -.5*log(negH);     % term for integrating laplace (denominator)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
