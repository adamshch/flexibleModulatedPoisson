%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Demo for fitting data using the flexible modeulated Poisson %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This demonstration shows how to fit the exponential, soft-rectification,
% and power-soft rectification models form the paper:
%
%   Charles, Park, Weller, Horowitz & Pillow. Dethroning the Fano Factor: A
%   Flexible, Model-Based Approach to Partitioning Neural Variability.
%   Neural Computation (2018).
%
% This script generates synthetic Poisson data with varying means,
% simulating orientation tuning with overdispersion. It then fits all three
% models and plots a comparison figure at the end. 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create some temporary simulated data

n_ori   = 70;                                                              % number of discrete stimuli
n_trial = 50;                                                              % number of trials for each stimuli

stims = linspace(0,10,n_ori); % stimuli

% Set nonlinearity and tuning curve
p = 2;                                                                     % Set exponent of the nonlinearity
if p == 1
    signse = 7;                                                            % Set the noise standard deviation
    ftune = sin((2*pi/10)*stims)*50+40;                                    % Generate the tuning curve to test
elseif p == 2
    signse = .66;                                                          % Set noise standard deviation
    ftune = sin((2*pi/10)*stims)*3+3;                                      % Generate the tuning curve to test
end
g = @(x)softrectpow(x,p);                                                  % Run data through the nonlinearity

Rtrain = poissrnd(g(repmat(ftune,n_trial,1)+randn(n_trial,n_ori)*signse)); % Generate training data

% Plot empirical tuning curve from training data
subplot(211); 
plot(stims,ftune);
title('true mus');

subplot(212);
plot(stims, Rtrain, 'ko'); hold on;
h = plot(stims, mean(Rtrain), stims, var(Rtrain), 'r--'); hold off;
legend(h,'mean', 'var');
xlabel('x');
title('training data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute ML fit by optimizing the negative log-likelihood

negLexp = @(x) negLfun_latentPoiss(x,@logli_latentPoiss_exp,Rtrain);       % Define negative log-likelihood (for an exponential nonlinearity)
negLsr  = @(x) negLfun_latentPoiss(x,@logli_latentPoiss_softrect,Rtrain);  % Define negative log-likelihood (for a soft-rectification nonlinearity)
negLsrp = @(x) negLfun_latentPoiss(x,@logli_latentPoiss_softrectpow,Rtrain);% Define negative log-likelihood (for a soft-rectification nonlinearity rased to a power)

prs0   = zeros(n_ori+1,1);                                                 % Initialize log-likelihood (for an exponential/softrect nonlinearity this is the number of orientations plus 1 variance parameter)
opts   = optimset('display','iter','largescale','off','maxfunevals',1e5);  % Set optimization paraemters for fminunc
prshat_exp = fminunc(negLexp,prs0,opts);                                   % Run fminunc to find the parameters (for an exponential nonlinearity)
prshat_sr  = fminunc(negLsr ,prs0,opts);                                   % Run fminunc to find the parameters (for a soft-rectification nonlinearity)
prs0   = zeros(n_ori+2,1);                                                 % Initialize log-likelihood (for an exponential/softrect nonlinearity this is the number of orientations plus 1 variance parameter and 1 power parameter)
prshat_srp = fminunc(negLsrp,prs0,opts);                                   % Run fminunc to find the parameters (for a soft-rectification nonlinearity rased to a power)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Make fig

mu_hat_exp  = prshat_exp(1:end-1);                                         % Extract optimized mean parameter (exponential)
mu_hat_sr   = prshat_sr(1:end-1);                                          % Extract optimized mean parameter (soft-rectification)
mu_hat_srp  = prshat_srp(1:end-2);                                         % Extract optimized mean parameter (power soft-rectification)
sig_hat_exp = exp(prshat_exp(end));                                        % Extract optimized variance parameter (exponential)
sig_hat_sr  = exp(prshat_sr(end));                                         % Extract optimized variance parameter (soft-rectification)
sig_hat_srp = exp(prshat_srp(end-1));                                      % Extract optimized variance parameter (power soft-rectification)
p_hat_srp   = exp(prshat_srp(end));                                        % Extract optimized variance parameter (power soft-rectification)
clf;
plot(stims,ftune,'b',stims,mu_hat_exp,'r',stims,mu_hat_sr,'g',stims,mu_hat_srp,'k','linewidth', 2);
legend('true', 'exp', 'soft-rect','soft-rect-p');
set(gca,'tickdir', 'out'); box off;
title(sprintf('(sig=%.2f,sig_{exp}=%.2f,sig_{sr}=%.2f,sig_{srp}=%.2f,p_{srp}=%1.1f)',signse,sig_hat_exp,sig_hat_sr,sig_hat_srp,p_hat_srp));
xlabel('x'); ylabel('mu');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
