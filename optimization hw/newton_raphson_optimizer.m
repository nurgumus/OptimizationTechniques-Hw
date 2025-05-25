function [xk, fk, gk_norm, k, path_hist] = newton_raphson_optimizer(f, grad_f, hess_f, x0, epsilon, max_iter)
    xk = x0;
    fk = f(xk);
    gk = grad_f(xk);
    gk_norm = norm(gk);
    path_hist = [xk]; % Store initial point

    fprintf('Iter %3d: f(x)=%e, ||grad||=%e\n', 0, fk, gk_norm);

    for k = 1:max_iter
        Hk = hess_f(xk);
        
        % Check for singularity / positive definiteness (optional, can use pinv)
        if rcond(Hk) < 1e-12 % Reciprocal condition number
           warning('Hessian is ill-conditioned or singular. Using pseudo-inverse.');
           pk = -pinv(Hk) * gk;
        else
           pk = -Hk \ gk; % Solve Hk * pk = -gk
        end

        % Simple line search (alpha=1 for pure Newton)
        alpha = 1; 
        % Add backtracking line search here for robustness if needed
        % [alpha, ~] = backtracking_line_search(f, grad_f, xk, pk, 1.0, 0.5, 1e-4);
        
        x_new = xk + alpha * pk;
        f_new = f(x_new);
        
        % Store path
        path_hist = [path_hist, x_new];

        % Stopping criteria
        if norm(grad_f(x_new)) <= epsilon && abs(f_new - fk) <= epsilon
            xk = x_new;
            fk = f_new;
            gk_norm = norm(grad_f(xk));
            fprintf('Iter %3d: f(x)=%e, ||grad||=%e. Converged.\n', k, fk, gk_norm);
            return;
        end
        
        xk = x_new;
        fk = f_new;
        gk = grad_f(xk);
        gk_norm = norm(gk);
        
        if mod(k,10)==0 || k < 5 % Print every 10 iterations or first few
            fprintf('Iter %3d: f(x)=%e, ||grad||=%e\n', k, fk, gk_norm);
        end
    end
    warning('Newton-Raphson: Maximum iterations reached without convergence.');
end