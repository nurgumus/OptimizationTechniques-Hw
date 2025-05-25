function [xk, fk, gk_norm, k_iter, path_hist] = hestenes_stiefel_optimizer(f, grad_f, x0, epsilon, max_iter, n_dim)
    xk = x0;
    fk = f(xk);
    gk = grad_f(xk);
    dk = -gk;
    gk_norm = norm(gk);
    path_hist = [xk];
    fprintf('Iter %3d: f(x)=%e, ||grad||=%e\n', 0, fk, gk_norm);

    for k_iter = 1:max_iter
        alpha_k = backtracking_line_search(f, grad_f, xk, dk, 1.0, 0.5, 1e-4);
        if alpha_k < 1e-8, warning('H-S: Line search step too small.'); break; end

        x_new = xk + alpha_k * dk;
        f_new = f(x_new);
        gk_new = grad_f(x_new);
        path_hist = [path_hist, x_new];

        if norm(gk_new) <= epsilon && abs(f_new - fk) <= epsilon
            xk = x_new; fk = f_new; gk_norm = norm(gk_new);
            fprintf('Iter %3d: f(x)=%e, ||grad||=%e. Converged.\n', k_iter, fk, gk_norm);
            return;
        end

        yk = gk_new - gk; % y_k = g_{k+1} - g_k
        denominator_hs = dk' * yk;
        if abs(denominator_hs) < 1e-10 % Avoid division by zero
            beta_k_plus_1 = 0; % Restart if denominator is too small
        else
            beta_k_plus_1 = (gk_new' * yk) / denominator_hs;
        end
        
        dk_new = -gk_new + beta_k_plus_1 * dk;

        if mod(k_iter, n_dim * 2) == 0 || (gk_new' * dk_new >= 0)
            dk_new = -gk_new;
        end
        
        xk = x_new; fk = f_new; gk = gk_new; dk = dk_new; gk_norm = norm(gk);
        if mod(k_iter,10)==0 || k_iter < 5
            fprintf('Iter %3d: f(x)=%e, ||grad||=%e\n', k_iter, fk, gk_norm);
        end
    end
    warning('Hestenes-Stiefel: Maximum iterations reached.');
end