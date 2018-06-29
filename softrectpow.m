function [f,df,ddf] = softrectpow(x,pow)

% [f,df,ddf] = softrectpow(x,pow)
%
% Computes the soft-rectified power function:
%
%     f(x) = log(1+exp(x)).^pow;
%
%  and its first (df) and second (ddf) derivatives, if desired. 
%  For very small or large values, numerical errors may hinder calculation. 
%  In the case of small values (x < -20), the approximation 
%
%     f(x)   = exp(pow*x)
%     df(x)  = pow*exp(pow*x)
%     ddf(x) = pow^2*exp(pow*x) 
%
%  is used. For very large values (x > 500) the approximation
%
%     f(x)   = x^pow
%     df(x)  = pow*x^(pow-1)
%     ddf(x) = pow*(pow-1)*x^(pow-2)
%
%  is used
%
%  Inputs:
%      x:   array of values to calculate f/df/ddf over 
%      pow: scalar value of the power for the soft rectification
%      
%  Outputs:
%      f:   array of values of f(x) that is the same size as x
%      df:  array of values of df(x) that is the same size as x
%      ddf: array of values of ddf(x) that is the same size as x
%      
%
% 2018 - Adam Charles & Jonathan Pillow


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f0 = log(1+exp(x));                                                        % Calculate base function value
f = f0.^pow;                                                               % Raise base to a power

if nargout > 1
    df = pow*f0.^(pow-1).*exp(x)./(1+exp(x));                              % Calculate the first derivative if needed
end
if nargout > 2
    if pow == 1
        ddf = pow*f0.^(pow-1).*exp(x)./(1+exp(x)).^2;                      % Calculate the second derivative if needed
    else
        ddf = pow*f0.^(pow-1).*exp(x)./(1+exp(x)).^2 + ...
              pow*(pow-1)*f0.^(pow-2).*(exp(x)./(1+exp(x))).^2;            % Second derivative is different for p != 1
    end
end

% Check for small values
if any(x(:)<-20)
    iix = (x(:)<-20);
    f(iix) = exp(pow*x(iix));
    df(iix) = pow*exp(pow*x(iix));
    ddf(iix) = pow^2*exp(pow*x(iix));
end

% Check for large values
if any(x(:)>500)
    iix = (x(:)>500);
    f(iix) = x(iix).^pow;
    df(iix) = pow*x(iix).^(pow-1);
    if pow == 1
        ddf(iix) = 0;
    else
        ddf(iix) = pow*(pow-1)*x(iix).^(pow-2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
