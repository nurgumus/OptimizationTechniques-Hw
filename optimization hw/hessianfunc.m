function H = hessianfunc(x, params)
% hessianfunc.m - Evaluates the Hessian matrix of the McCormick function f_24(x)
% INPUTS:
%   x - 2x1 column vector of design variables [x1; x2]
%   params - struct containing function parameters (not used here)
% OUTPUT:
%   H - 2x2 Hessian matrix at x

% Ensure x is a column vector
if size(x,2) > 1; x = x'; end

if length(x) ~= 2
    error('Hessian of McCormick function (f_24) expects a 2D vector x.');
end

x1 = x(1);
x2 = x(2);

H = zeros(2,2);
common_term = -sin(x1 + x2);

H(1,1) = common_term + 2;
H(1,2) = common_term - 2;
H(2,1) = H(1,2); % Hessian is symmetric
H(2,2) = common_term + 2;

end