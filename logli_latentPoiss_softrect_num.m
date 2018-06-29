function logL = logli_latentPoiss_softrect_num(r,mu,sig,nstds,nbins)

% logL = logli_latentPoiss_softrect_num(r,mu,sig,nstds,nbins)
%
% Numerically compute loglikelihoods under latent-gaussian Poisson model
% with soft-rectifying nonlinearity 
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

if nargin < 5
    nbins = 1e4;                                                           % Set the number of bins if not already provided
end
if nargin < 3
    nstds = 15;                                                            % Set the number of standard deviations to numerically integrate over
end

% make mu and sigma the same size as mu
rsz = size(r);
if length(mu)==1
    mu = repmat(mu,rsz);
end
if length(sig)==1
    sig = repmat(sig,rsz);
end

x    = linspace(-nstds,nstds,nbins);                                       % Set up the grid points to numerically integrate over
dx   = diff(x(1:2));                                                       % Get the bin width (distance between points)
logL = zeros(rsz);                                                         % Initialize the array to store the per-count log-likelihood values

for jj = 1:numel(r)
    logL(jj) = log(sum(normpdf(x*sig(jj),0,sig(jj)).*poisspdf(r(jj),...
                                 softrect(x*sig(jj)+mu(jj))))*dx*sig(jj)); % Iteratively calculate the per-count log-likelihood
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
