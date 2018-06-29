function [f,df,ddf] = softrect2(x)
%  [f,df,ddf] = logexp_pow(x);
%
%  Computes the squared soft-rectifying nonlinearity:  
%     f(x) = log(1+exp(x)).^2
%  antd first and second derivatives.
%  For very small or large values, numerical errors may hinder calculation. 
%  In the case of small values (x < -20), the approximation 
%     f(x) = exp(2x), df(x) = 2exp(2x), ddf(x) = 4exp(2x) 
%  is used. For very large values (x > 500) the approximation
%     f(x) = x^2, df(x) = 2x, ddf(x) = 2
%  is used
%
%  Inputs:
%      x: array of values to calculate f/df/ddf over 
%      
%  Outputs:
%      f:   array of values of f(x) that is the same size as x
%      df:  array of values of df(x) that is the same size as x
%      ddf: array of values of ddf(x) that is the same size as x
%      
%
% 2018 - Adam Charles & Jonathan Pillow


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


pow = 2;                                                                   % Set the power to 2
f0  = log(1+exp(x));                                                       % Calculate the base of the function
f   = f0.^pow;                                                             % Square the base

if nargout > 1
    df = pow*f0.^(pow-1).*exp(x)./(1+exp(x));                              % If necessary output the first derivatives
end
if nargout > 2
    ddf = pow*f0.^(pow-1).*exp(x)./(1+exp(x)).^2 + ...
          pow*(pow-1)*f0.^(pow-2).*(exp(x)./(1+exp(x))).^2;                % If necessary output the second derivatives
end

% Check for small values
if any(x(:)<-20)
    iix = (x(:)<-20);
    f(iix) = exp(2*x(iix));
    df(iix) = 2*f(iix);
    ddf(iix) = 4*f(iix);
end

% Check for large values
if any(x(:)>500)
    iix = (x(:)>500);
    f(iix) = x(iix).^2;
    df(iix) = 2*x(iix);
    ddf(iix) = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
