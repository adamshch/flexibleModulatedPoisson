%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Demo for fitting data using the flexible modeulated Poisson %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
h = plot(stims mean(Rtrain), stims, var(Rtrain), 'r--'); hold off;
legend(h,'mean', 'var');
xlabel('x');
title('training data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute ML fit by optimizing the negative log-likelihood

negLexp = @(x) un_latentPoiss(x,@logli_latentPoiss_exp,Rtrain);            % Define negative log-likelihood (for an exponential nonlinearity)
negLsr  = @(x) un_latentPoiss(x,@logli_latentPoiss_softrect,Rtrain);       % Define negative log-likelihood (for a soft-rectification nonlinearity)
negLsrp = @(x) un_latentPoiss(x,@logli_latentPoiss_softrectpow,Rtrain);    % Define negative log-likelihood (for a soft-rectification nonlinearity rased to a power)

prs0   = zeros(n_ori+1,1);                                                 % Initialize log-likelihood
opts   = optimset('display','iter','largescale','off','maxfunevals',1e5);  % Set optimization paraemters for fminunc
prshat = fminunc(negLfun,prs0,opts);                                       % Run fminunc to find the parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Make fig

mu_hat  = prshat(1:end-1);                                                 % Extract optimized mean parameter
sig_hat = exp(prshat(end));                                                % Extract optimized variance parameter
clf;
plot(stims,ftune,stims,muhat,'linewidth', 2);
legend('true', 'estimate');
set(gca,'tickdir', 'out'); box off;
title(sprintf('(sig=%.2f,sighat=%.2f)',signse,sighat));
xlabel('x'); ylabel('mu');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
