function g = gradfunc(x, params)
% gradfunc.m - Evaluates the gradient of the McCormick function f_24(x)
% INPUTS:
%   x - 2x1 column vector of design variables [x1; x2]
%   params - struct containing function parameters (not used here)
% OUTPUT:
%   g - 2x1 column vector of the gradient at x [df/dx1; df/dx2]

% Ensure x is a column vector
if size(x,2) > 1; x = x'; end

if length(x) ~= 2
    error('Gradient of McCormick function (f_24) expects a 2D vector x.');
end

x1 = x(1);
x2 = x(2);

g = zeros(2,1);

g(1) = cos(x1 + x2) + 2*(x1 - x2) - 1.5;
g(2) = cos(x1 + x2) - 2*(x1 - x2) + 2.5;

end