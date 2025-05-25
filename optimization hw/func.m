function val = func(x, params)
% func.m - Evaluates the McCormick objective function f_24(x)
% because my student number is 21052083 and cd = 24 (mod 50)
% INPUTS:
%   x - 2x1 column vector of design variables [x1; x2]
%   params - struct containing function parameters (not used for this specific function,
%            but kept for consistency with the optimizer framework)
% OUTPUT:
%   val - scalar value of the function at x

% Ensure x is a column vector
if size(x,2) > 1; x = x'; end % Transpose if it's a row vector

if length(x) ~= 2
    error('McCormick function (f_24) expects a 2D vector x.');
end

x1 = x(1);
x2 = x(2);

val = sin(x1 + x2) + (x1 - x2)^2 - 1.5*x1 + 2.5*x2 + 1;

% The parameter 'params.cd_value' could be checked here if you had multiple
% functions within this one file, but since it's dedicated to f_24,
% it's not strictly necessary.
% if params.cd_value ~= 24
%     warning('func.m called with unexpected cd_value in params.');
% end

end