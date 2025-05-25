% --- 1. Parameter Selection ---
% my student number is 21052083
a = 8; 
b = 3; 
cd_value = mod(a * b, 50); % This will be 24

fprintf('Student parameters: a = %d, b = %d\n', a, b);
fprintf('Calculated cd = %d (McCormick Function)\n\n', cd_value);

% --- 2. Problem Definition & Algorithm Parameters ---
problem_params.cd_value = cd_value; % Store cd_value if needed by generic functions
problem_params.n_dim = 2;           % Dimension of the McCormick problem
problem_params.x0_min = [-1.5; -3];  % Lower bounds for x1, x2 (COLUMN VECTOR)
problem_params.x0_max = [4; 3];      % Upper bounds for x1, x2 (COLUMN VECTOR)

% --- Function Handles ---
% The params argument is kept for consistency, even if func/grad/hess for McCormick don't use it.
f = @(x) func(x, problem_params);
grad_f = @(x) gradfunc(x, problem_params);
hess_f = @(x) hessianfunc(x, problem_params);

% --- Algorithm Settings ---
epsilon = 1e-4;
max_iterations = 1000; % Max iterations to prevent infinite loops

% --- 3. Initial Guesses ---
num_initial_guesses = 3;
initial_guesses = cell(num_initial_guesses, 1);

rng('default'); % For reproducibility
fprintf('Generating %d initial guesses...\n', num_initial_guesses);
for i = 1:num_initial_guesses
    % Option 1: Normal distribution
    % initial_guesses{i} = randn(problem_params.n_dim, 1);

    % Option 2: Uniform distribution (use if bounds are well-defined for f_cd)
    % Ensure x0_min and x0_max are column vectors of size n_dim x 1
    initial_guesses{i} = problem_params.x0_min + ...
        (problem_params.x0_max - problem_params.x0_min) .* rand(problem_params.n_dim, 1);
    fprintf('Initial Guess %d: [%s]\n', i, sprintf('%.4f ', initial_guesses{i}));
end
fprintf('\n');

% --- 4. Algorithms to Test ---
algorithms = {
    'Newton-Raphson', @newton_raphson_optimizer;
    'Hestenes-Stiefel', @hestenes_stiefel_optimizer;
    'Polak-Ribiere', @polak_ribiere_optimizer;
    'Fletcher-Reeves', @fletcher_reeves_optimizer;
    'Dai-Yuan', @dai_yuan_optimizer 
};
num_algorithms = size(algorithms, 1);

% --- 5. Results Storage ---
results = cell(num_initial_guesses, num_algorithms);
% Each cell will store a struct: {x_final, f_final, grad_norm_final, iterations, time_taken, path}

% --- 6. Run Optimizers ---
for i = 1:num_initial_guesses
    x0 = initial_guesses{i};
    fprintf('--- Processing Initial Guess %d: [%s] ---\n', i, sprintf('%.4f ', x0));
    for j = 1:num_algorithms
        algo_name = algorithms{j, 1};
        algo_func = algorithms{j, 2};

        fprintf('Running %s...\n', algo_name);
        tic;
        if strcmp(algo_name, 'Newton-Raphson')
            [x_final, f_final, grad_norm_final, k, path_hist] = ...
                algo_func(f, grad_f, hess_f, x0, epsilon, max_iterations);
        else % Conjugate Gradient methods
            [x_final, f_final, grad_norm_final, k, path_hist] = ...
                algo_func(f, grad_f, x0, epsilon, max_iterations, problem_params.n_dim);
        end
        time_taken = toc;

        results{i, j} = struct(...
            'x_final', x_final, ...
            'f_final', f_final, ...
            'grad_norm_final', grad_norm_final, ...
            'iterations', k, ...
            'time_taken', time_taken, ...
            'path', path_hist ...
        );

        fprintf('%s: Iterations = %d, f(x*) = %.4e, ||grad(f(x*))|| = %.4e, Time = %.4f s\n', ...
                algo_name, k, f_final, grad_norm_final, time_taken);
        fprintf('   x* = [%s]\n', sprintf('%.4f ', x_final));
    end
    fprintf('\n');
end


% --- 8. Plotting ---
if problem_params.n_dim == 2
    fprintf('Generating 2D plots...\n');
    figure;
    hold on;
    grid on;

    % Contour plot of the function
    x1_plot = linspace(problem_params.x0_min(1)-0.5, problem_params.x0_max(1)+0.5, 100);
    x2_plot = linspace(problem_params.x0_min(2)-0.5, problem_params.x0_max(2)+0.5, 100);
    [X1, X2] = meshgrid(x1_plot, x2_plot);
    Z = zeros(size(X1));
    for r = 1:size(X1,1)
        for c = 1:size(X1,2)
            Z(r,c) = f([X1(r,c); X2(r,c)]);
        end
    end
    contour(X1, X2, Z, 50); % Adjust number of contour lines as needed
    colorbar;
    title(sprintf('Optimization Paths for f_{%d}(x) (2D Example)', cd_value));
    xlabel('x_1');
    ylabel('x_2');

    % Colors for initial guesses
    initial_guess_colors = lines(num_initial_guesses); % Different color for each initial guess path



    % Plot paths for a selected algorithm (e.g., Newton-Raphson, index 1)
    selected_algo_idx_for_iter_plot = 5; % change here for every graph!!!!!!!!!1**
    legend_entries = {};
    legend_handles = [];

    % Plot all paths, colored by initial guess
    for i = 1:num_initial_guesses % Loop through initial guesses
        path_color = initial_guess_colors(i,:);
        plot(initial_guesses{i}(1), initial_guesses{i}(2), 'o', ...
            'MarkerFaceColor', path_color, 'MarkerEdgeColor', 'k', 'DisplayName', sprintf('Start %d', i));
        legend_entries{end+1} = sprintf('Start %d', i);
        legend_handles(end+1) = plot(NaN,NaN,'o','MarkerFaceColor', path_color, 'MarkerEdgeColor', 'k'); % For legend

        for j = 1:num_algorithms % Loop through algorithms
            algo_name = algorithms{j,1};
            path_data = results{i,j}.path;
            line_style = '-'; % Default
            marker_char = '.';
            if j == 1, line_style = '-'; marker_char = '.'; end % NR
            if j == 2, line_style = '--'; marker_char = 'x'; end % HS
            if j == 3, line_style = ':'; marker_char = '+'; end % PR
            if j == 4, line_style = '-.'; marker_char = 's'; end % FR
            if j == 5, line_style = '-'; marker_char = 'd'; end % DY (diamond marker)

            if ~isempty(path_data)
                h = plot(path_data(1,:), path_data(2,:), [marker_char line_style], ...
                    'Color', path_color, ...
                    'LineWidth', 1.5, ...
                    'MarkerSize', 6);
                if i == 1 % Add algorithm to legend only once
                    legend_entries{end+1} = algo_name;
                    legend_handles(end+1) = h;
                end
            end
        end
    end
    legend(legend_handles, legend_entries, 'Location', 'northeastoutside');
    hold off;

    % "Same iteration, same color" plot for ONE algorithm (e.g., Newton)
    figure;
    hold on; grid on;
    contour(X1, X2, Z, 50); colorbar;
    title(sprintf('%s: Iteration Steps (Same Iteration = Same Color)', algorithms{selected_algo_idx_for_iter_plot,1}));
    xlabel('x_1'); ylabel('x_2');
    
    max_iter_this_algo = 0;
    paths_this_algo = cell(num_initial_guesses,1);
    for i=1:num_initial_guesses
        paths_this_algo{i} = results{i, selected_algo_idx_for_iter_plot}.path;
        if ~isempty(paths_this_algo{i})
            max_iter_this_algo = max(max_iter_this_algo, size(paths_this_algo{i},2));
        end
    end

    iter_colors = cool(max_iter_this_algo); % Different color for each iteration number

    for i=1:num_initial_guesses % Plot initial points first
        plot(initial_guesses{i}(1), initial_guesses{i}(2), 'o', ...
            'MarkerSize', 8, 'MarkerFaceColor', initial_guess_colors(i,:), 'MarkerEdgeColor', 'k', ...
            'DisplayName', sprintf('Start %d',i));
    end

    for k_iter = 1:max_iter_this_algo % Loop through iteration steps
        current_iter_color = iter_colors(k_iter,:);
        for i = 1:num_initial_guesses % Loop through initial guesses
            path_data = paths_this_algo{i};
            if ~isempty(path_data) && k_iter <= size(path_data,2)
                % Plot current iteration point
                plot(path_data(1,k_iter), path_data(2,k_iter), '.', ...
                    'Color', current_iter_color, 'MarkerSize', 15);
                % Plot line from previous point (if it exists) using initial guess color
                if k_iter > 1 && k_iter <= size(path_data,2)
                     plot([path_data(1,k_iter-1), path_data(1,k_iter)], ...
                          [path_data(2,k_iter-1), path_data(2,k_iter)], ...
                          '-', 'Color', initial_guess_colors(i,:), 'LineWidth', 0.5);
                elseif k_iter == 1 % Line from x0 to x1
                     plot([initial_guesses{i}(1), path_data(1,k_iter)], ...
                          [initial_guesses{i}(2), path_data(2,k_iter)], ...
                          '-', 'Color', initial_guess_colors(i,:), 'LineWidth', 0.5);
                end
            end
        end
    end
    legend('show','Location','northeastoutside');
    hold off;


elseif problem_params.n_dim > 2
    fprintf('Generating N-D plots (f(x_k) and ||grad(f(x_k))|| vs iterations)...\n');
    figure; % Plot for function values
    subplot(2,1,1); hold on; grid on;
    title(sprintf('Function Value f(x_k) vs Iterations (f_{%d})', cd_value));
    xlabel('Iteration k'); ylabel('f(x_k)');
    
    subplot(2,1,2); hold on; grid on;
    title(sprintf('Gradient Norm ||grad(f(x_k))|| vs Iterations (f_{%d})', cd_value));
    xlabel('Iteration k'); ylabel('||grad(f(x_k))||');
    
    legend_entries_nd = {};
    plot_handles_fval = [];
    plot_handles_grad = [];

    line_styles = {'-', '--', ':', '-.', '-'}; % For different algorithms

    for i = 1:num_initial_guesses % Consider plotting average or for one initial guess
        if i > 1; continue; end % For N-D, plot for first initial guess for clarity
        fprintf('Plotting for initial guess %d\n',i);
        
        for j = 1:num_algorithms
            algo_name = algorithms{j,1};
            path_data = results{i,j}.path;
            if isempty(path_data); continue; end
            
            num_iters_path = size(path_data, 2);
            f_values = zeros(1, num_iters_path);
            grad_norm_values = zeros(1, num_iters_path);
            
            for k_path = 1:num_iters_path
                f_values(k_path) = f(path_data(:, k_path));
                grad_norm_values(k_path) = norm(grad_f(path_data(:, k_path)));
            end
            
            current_line_style = line_styles{mod(j-1, length(line_styles))+1};

            subplot(2,1,1);
            h_fval = plot(0:num_iters_path-1, f_values, current_line_style, 'LineWidth', 1.5);
            if i==1, plot_handles_fval(end+1) = h_fval; end


            subplot(2,1,2);
            h_grad = plot(0:num_iters_path-1, grad_norm_values, current_line_style, 'LineWidth', 1.5);
            if i==1, plot_handles_grad(end+1) = h_grad; end
            
            if i==1, legend_entries_nd{end+1} = algo_name; end
        end
    end
    subplot(2,1,1); legend(plot_handles_fval, legend_entries_nd, 'Location', 'northeast'); set(gca, 'YScale', 'log');
    subplot(2,1,2); legend(plot_handles_grad, legend_entries_nd, 'Location', 'northeast'); set(gca, 'YScale', 'log');
end

% --- 9. Table for Benchmark ---
fprintf('\n--- Benchmark Table ---\n');
fprintf('%-10s | %-20s | %-10s | %-12s | %-15s | %-10s | %-20s\n', ...
    'InitGuess', 'Algorithm', 'Iterations', 'f(x*)', '||grad(f(x*))||', 'Time (s)', 'x*');
fprintf([repmat('-',1,110) '\n']);

for i = 1:num_initial_guesses
    for j = 1:num_algorithms
        res = results{i,j};
        x_star_str = sprintf('%.4f ', res.x_final);
        fprintf('%-10d | %-20s | %-10d | %-12.4e | %-15.4e | %-10.4f | %s\n', ...
            i, algorithms{j,1}, res.iterations, res.f_final, res.grad_norm_final, res.time_taken, x_star_str);
    end
    if i < num_initial_guesses
        fprintf([repmat('-',1,110) '\n']);
    end
end
fprintf('\nNOTE: Use 4 significant figures AFTER the decimal point in your report for numerical values.\n');
fprintf('Code uses scientific notation for f(x*) and ||grad|| for compactness here.\n');

% --- End of Script ---

% In the "Stationary Point Analysis" section of Nonlinear.m, you can add:
fprintf('Known global minimum for McCormick (f_24) is approx at x* = [-0.547; -1.547] with f(x*) approx -1.9133\n');
fprintf('Known local minimum for McCormick (f_24) is approx at x_local* = [2.59; 1.59]\n');
% You can then try to use fsolve on your gradfunc to see if it finds these:
% options_fsolve = optimoptions('fsolve', 'Display','iter', 'FunctionTolerance', 1e-8, 'StepTolerance', 1e-8);
% grad_f_anon = @(x) gradfunc(x, problem_params); % Make sure gradfunc can be called with x only
%
% fprintf('Attempting to find stationary point near global minimum guess...\n');
% [x_stat_glob, fval_glob, exitflag_glob, output_glob] = fsolve(grad_f_anon, [-0.5; -1.5], options_fsolve);
% if exitflag_glob > 0
%     fprintf('fsolve found point: [%.4f, %.4f], grad norm: %.2e, f(x)=%.4f\n', x_stat_glob(1), x_stat_glob(2), norm(grad_f(x_stat_glob)), f(x_stat_glob));
%     H_stat_glob = hess_f(x_stat_glob);
%     eig_H_glob = eig(H_stat_glob);
%     fprintf('Eigenvalues of Hessian at this point: [%s]\n', sprintf('%.4f ', eig_H_glob));
%     if all(eig_H_glob > 0), fprintf('Classification: Local Minimum\n'); end
% end
%
% fprintf('Attempting to find stationary point near local minimum guess...\n');
% [x_stat_loc, fval_loc, exitflag_loc, output_loc] = fsolve(grad_f_anon, [2.5; 1.5], options_fsolve);
% if exitflag_loc > 0
%     fprintf('fsolve found point: [%.4f, %.4f], grad norm: %.2e, f(x)=%.4f\n', x_stat_loc(1), x_stat_loc(2), norm(grad_f(x_stat_loc)), f(x_stat_loc));
%     H_stat_loc = hess_f(x_stat_loc);
%     eig_H_loc = eig(H_stat_loc);
%     fprintf('Eigenvalues of Hessian at this point: [%s]\n', sprintf('%.4f ', eig_H_loc));
%     if all(eig_H_loc > 0), fprintf('Classification: Local Minimum\n'); end
% end