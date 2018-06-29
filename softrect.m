function [f,df,ddf] = softrect(x)

%  [f,df,ddf] = softrect(x);
%
%  Computes the soft-rectification function
%  
%     f(x) = log(1+exp(x))
%
%  and its first (df) and second (ddf) derivatives, if desired. 
%  For very small or large values, numerical errors may hinder calculation. 
%  In the case of small values (x < -20), the approximation 
%     f(x) = df(x) = ddf(x) = exp(x) 
%  is used. For very large values (x > 500) the approximation
%     f(x) = x, df(x) = 1, ddf(x) = 0
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

f = log(1+exp(x));                                                         % Calculate the function values

if nargout > 1
    df = exp(x)./(1+exp(x));                                               % If needed calculate the first derivative
end
if nargout > 2
    ddf = exp(x)./(1+exp(x)).^2;                                           % If needed calculate the second derivative
end

% Check for small values
if any(x(:)<-20)
    iix = (x(:)<-20);
    f(iix) = exp(x(iix));
    df(iix) = f(iix);
    ddf(iix) = f(iix);
end

% Check for large values
if any(x(:)>500)
    iix = (x(:)>500);
    f(iix) = x(iix);
    df(iix) = 1;
    ddf(iix) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
