function negL = negLfun_latentPoiss(prs,loglifun,Rtrain)

% function negL = negLfun_latentPoiss(prs,loglifun,Rtrain)
%
% This funcition computes the negative-log likelihood of a matrix of 
% spike counts, given a set of parameters, using the Laplace 
% approximation.  The intended use of this function is in conjunction 
% with fmincon (or a similar optimization program) in order to find the 
% parameters that produce the maximum log-likelihood (minimum negative
% log-likelihood). 
%
% Inputs: 
%       prs - parameter vector
%  loglifun - anonymous function handle to negative log-likelihood function 
%             (e.g. @logli_latentPoiss_exp)
%    Rtrain - [nsamps x noris] matrix of data
%
% Output:
%    negL - negative log-likelihood
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nsamps,nstim] = size(Rtrain);                                             % number of repeats per stimiulus, number of stimuli

% extract params
mus      = prs(1:nstim)';                                                  % mean params
otherprs = exp(prs(nstim+1:end));                                          % exponentiate other params so they stay positive

% Compute neglogli for each response
negLvals = loglifun(Rtrain,repmat(mus,nsamps,1),otherprs);
negL     = -sum(negLvals(:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
