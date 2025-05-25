function [alpha, f_new] = backtracking_line_search(f, grad_f, xk, pk, alpha_init, rho, c)
% backtracking_line_search - Finds a step size alpha satisfying Armijo condition.
% INPUTS:
%   f        - function handle for objective function
%   grad_f   - function handle for gradient
%   xk       - current point
%   pk       - search direction
%   alpha_init - initial step size (e.g., 1.0)
%   rho      - reduction factor for alpha (e.g., 0.5)
%   c        - Armijo condition constant (e.g., 1e-4)
% OUTPUTS:
%   alpha    - step size satisfying Armijo condition
%   f_new    - function value at xk + alpha*pk

    alpha = alpha_init;
    fk = f(xk);
    gk_pk = grad_f(xk)' * pk; % Gradient projected onto search direction

    max_ls_iters = 50; % Max iterations for line search
    ls_iter = 0;
    
    while ls_iter < max_ls_iters
        x_new = xk + alpha * pk;
        f_new = f(x_new);
        if f_new <= fk + c * alpha * gk_pk % Armijo condition
            return;
        end
        alpha = rho * alpha;
        ls_iter = ls_iter + 1;
        if alpha < 1e-10 % Avoid excessively small alpha
            % warning('Line search: Alpha became too small.');
            break; 
        end
    end
    % If Armijo not satisfied after max_ls_iters, return current (small) alpha
    % This might happen if pk is not a descent direction or c is too large
    % or numerical precision issues.
    if ls_iter == max_ls_iters
        % warning('Line search: Max iterations reached. Armijo condition might not be strongly satisfied.');
    end
end