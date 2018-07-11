function x = softrectpow_inv(y,pow)
% x = softrectpow_inv(y,pow)
%
% Computes the inverse of soft-rectification power function:
%     f(x) = log(1+exp(x)).^pow
% which is given by:
%     fi(y) = log(exp(y.^(1/pow))-1))
%  For very large values of y^(1./pow) (>50), numerical errors may hinder 
%  calculation and the approximation
%     fi(y) = y
%  is used.
%
%  Inputs:
%      y: array of values to calculate fi(y) over
%      
%  Outputs:
%      x:   array of values of fi(y) that is the same size as y
%      
%
% 2018 - Adam Charles & Jonathan Pillow


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ypow = y.^(1/pow);
x = log(exp(ypow)-1);

% Check for large values
if any(ypow(:)>50)
    iibig = (ypow>50);
    x(iibig) = ypow(iibig);
end    
