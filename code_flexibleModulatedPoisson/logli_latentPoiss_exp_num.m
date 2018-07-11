function logL = logli_latentPoiss_exp_num(r,mu,sig,nstds,nbins)

% logL = logli_latentPoiss_exp_num(r,mu,sig,nstds,nbins)
% 
% Numerically compute loglikelihoods under latent-gaussian Poisson model
% with exponential nonlinearity
%
% Inputs: 
%  r   -  spike count vector or matrix
%  mu  - mean (scalar or size-r matrix)
%  sig - std deviation (scalar or size-r matrix)
%
% 2018 - Jonathan Pillow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 5
    nbins = 1e4;                                                           % Set the number of bins if not already provided

end
if nargin < 4
    nstds = 10;                                                            % Set the number of standard deviations to numerically integrate over
end

% make mu and sigma the same size as mu
rsz = size(r);
if length(mu)==1
    mu = repmat(mu,rsz);
end
if length(sig)==1
    sig = repmat(sig,rsz);
end

x    = linspace(-nstds,nstds,nbins);                                       % Create a set of points to numerically calculate the log-liklihood over
dx   = diff(x(1:2));                                                       % Get the point separation distance between points
logL = zeros(rsz);                                                         % Initialize an array to store the log-likelihood per count

for jj = 1:numel(r)
    logL(jj) = log(sum(normpdf(x*sig(jj),0,sig(jj)).*poisspdf(r(jj),...
                                      exp(x*sig(jj)+mu(jj))))*dx*sig(jj)); % Iteratively calculate the log-likelihood per bin
end   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
