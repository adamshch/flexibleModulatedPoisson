function logL = logli_latentPoiss_exp(r,mu,sig)

% logL = logli_latentPoiss_exp(r,mu,sig)
%
% Compute loglikelihoods under latent-gaussian Poisson model
%
% Inputs: 
%  r   -  spike count vector or matrix
%  mu  - mean (scalar or size-r matrix)
%  sig - std deviation (scalar or size-r matrix)
% 
% Output: 
%  logL - log P(r | mu, sig) = \int P(r,x|mu,sig) dx
%
% 2018 - Jonathan Pillow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make mu and sigma the same size as mu

rsz = size(r);                                                             % Get the size of the data (number of spike counts)
if length(mu)==1
    mu = repmat(mu,rsz);                                                   % Make sure that the mean is a vector the same size as the data 
end
if length(sig)==1
    sig = repmat(sig,rsz);                                                 % Make sure that the variance is a vector the same size as the data 

end

% Compute Laplace approximation to posterior
[xmap,negH] = compLApprox_latentPoiss_exp(r,mu,sig);

% Compute integral using laplace approximation
logL = -log(sig)-xmap.^2./(2*sig.^2) + ...                                % prior Gaussian term
            + r.*(mu+xmap) - exp(mu+xmap) - gammaln(r+1) ...              % Poisson likelihood term
                                            -log(sqrt(negH));             % term for integrating laplace (denominator)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
