function logL = logli_latentPoiss_softrectpow(r,mu,sig,pow)
% logL = logli_latentPoiss_softrectpow(r,mu,sig,pow)
%
% Compute loglikelihoods under latent-gaussian Poisson model with
% soft-rectifying power function nonlinearity ('softrectpow').
%
% Inputs: 
%  r  -  spike count vector or matrix
%  mu - mean (scalar or size-r matrix)
%  sig - std deviation (scalar or size-r matrix)
%  pow - power of exponent (e.g., 1 or 2)
%
% Output: 
%  logL - log P(r | mu, sig) = \int P(r,x|mu,sig) dx


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% make mu and sigma the same size as mu
rsz = size(r);
if length(mu)==1,  mu  = repmat(mu,rsz);  end
if length(sig)==1, sig = repmat(sig,rsz); end

% Compute Laplace approximation to posterior
[xmap,negH] = compLApprox_latentPoiss_softrectpow(r,mu,sig,pow);

% Compute integral using laplace approximation
f    = softrectpow(mu+xmap,pow);                                           % Pre-calculate the soft-rectification raised to a power
logL = -log(sig)-xmap.^2./(2*sig.^2) + ...                                 % prior Gaussian term
                                + r.*log(f) - f -gammaln(r+1) + ...        % Poisson likelihood term
                                                            -.5*log(negH); % term for integrating laplace (denominator)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
