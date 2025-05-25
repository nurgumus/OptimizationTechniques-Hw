function [xk, fk, gk_norm, k_iter, path_hist] = dai_yuan_optimizer(f, grad_f, x0, epsilon, max_iter, n_dim)
% dai_yuan_optimizer.m - Implements the Dai-Yuan Conjugate Gradient method.
% INPUTS:
%   f        - function handle for objective function
%   grad_f   - function handle for gradient
%   x0       - initial guess (column vector)
%   epsilon  - tolerance for stopping criteria
%   max_iter - maximum number of iterations
%   n_dim    - dimension of the problem (used for potential restarts)
% OUTPUTS:
%   xk       - final point
%   fk       - function value at xk
%   gk_norm  - norm of the gradient at xk
%   k_iter   - number of iterations performed
%   path_hist- history of points (xk at each iteration)

    xk = x0;
    fk = f(xk);
    gk = grad_f(xk);    % g_k
    dk = -gk;           % d_k (initial direction is steepest descent)
    gk_norm = norm(gk);
    path_hist = [xk];   % Store initial point

    fprintf('Iter %3d (DY): f(x)=%e, ||grad||=%e\n', 0, fk, gk_norm);

    for k_iter = 1:max_iter
        % --- Line Search ---
        % Using the existing backtracking line search satisfying Armijo.
        % A line search satisfying Wolfe conditions is theoretically preferred for DY.
        alpha_k = backtracking_line_search(f, grad_f, xk, dk, 1.0, 0.5, 1e-4); % alpha_init, rho, c

        if alpha_k < 1e-10 % Step size too small, likely stuck or dk is poor
            warning('Dai-Yuan: Line search step size alpha_k is very small (%.2e). Stopping.', alpha_k);
            break;
        end

        % --- Update Point and Gradient ---
        x_new = xk + alpha_k * dk;  % x_{k+1}
        f_new = f(x_new);
        gk_new = grad_f(x_new);     % g_{k+1}
        
        path_hist = [path_hist, x_new]; % Store path

        % --- Stopping Criteria ---
        gk_new_norm = norm(gk_new);
        if gk_new_norm <= epsilon && abs(f_new - fk) <= epsilon
            xk = x_new;
            fk = f_new;
            gk_norm = gk_new_norm;
            fprintf('Iter %3d (DY): f(x)=%e, ||grad||=%e. Converged.\n', k_iter, fk, gk_norm);
            return;
        end

        % --- Calculate Dai-Yuan Beta (β_{k+1}) ---
        yk = gk_new - gk;       % y_k = g_{k+1} - g_k
        dk_T_yk = dk' * yk;     % Denominator term: d_k^T * y_k

        if abs(dk_T_yk) < 1e-12 % Denominator is close to zero, risk of instability
            % fprintf('Dai-Yuan: Denominator dk_T_yk (%.2e) is small. Restarting CG.\n', dk_T_yk);
            beta_k_plus_1 = 0; % Restart: effectively steepest descent for next step
        else
            beta_k_plus_1 = (gk_new' * gk_new) / dk_T_yk; % or norm(gk_new)^2 / dk_T_yk
        end
        
        % --- Compute New Search Direction ---
        % d_{k+1} = -g_{k+1} + β_{k+1} * d_k
        dk_new = -gk_new + beta_k_plus_1 * dk;

        % --- Optional Restart Conditions (besides small denominator) ---
        % Standard restart: e.g., every n_dim iterations or if not a descent direction
        % (gk_new' * dk_new >= 0 indicates dk_new is not a descent direction for g_new)
        if mod(k_iter, n_dim * 2) == 0 || (gk_new' * dk_new >= -1e-8 * norm(gk_new) * norm(dk_new) ) % Allow for slight non-descent due to precision
            if (gk_new' * dk_new >= -1e-8 * norm(gk_new) * norm(dk_new) ) && beta_k_plus_1 ~= 0
                 % fprintf('Dai-Yuan: New direction not sufficiently descent (g_new^T * d_new = %.2e). Restarting.\n', gk_new' * dk_new);
            end
            dk_new = -gk_new; % Restart to steepest descent
            % beta_k_plus_1 = 0; % Not strictly needed here as dk_new is reset
        end
        
        % --- Update variables for next iteration ---
        xk = x_new;
        fk = f_new;
        gk = gk_new;
        dk = dk_new;
        gk_norm = gk_new_norm; % Use the norm calculated earlier for gk_new

        if mod(k_iter,10)==0 || k_iter < 5 % Print progress
            fprintf('Iter %3d (DY): f(x)=%e, ||grad||=%e, beta=%.2e\n', k_iter, fk, gk_norm, beta_k_plus_1);
        end
    end

    warning('Dai-Yuan: Maximum iterations (%d) reached without convergence.', max_iter);
end