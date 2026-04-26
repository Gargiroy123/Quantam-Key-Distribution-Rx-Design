clc; clear; close all;

%% 🔷 PARAMETERS
M = 2;                 % BPSK
alpha = 1;           % amplitude
m_max = 6;             % photon count truncation

%% 🔷 BPSK STATES
alpha_k = [alpha, -alpha];

%% 🔷 PRIOR (Uniform)
P = [0.5, 0.5];

%% 🔷 BETA GRID (REAL AXIS)
beta_real = linspace(-3,3,300);
entropy_vals = zeros(size(beta_real));

%% 🔷 ENTROPY COMPUTATION
for b = 1:length(beta_real)
    
    beta = beta_real(b);   % real only
    expected_entropy = 0;
    
    for m = 0:m_max
        
        P_m = 0;
        likelihoods = zeros(1,M);
        
        % Likelihood for each hypothesis
        for j = 1:M
            lambda = abs(alpha_k(j) - beta)^2;
            likelihoods(j) = (lambda^m / factorial(m)) * exp(-lambda);
            P_m = P_m + P(j) * likelihoods(j);
        end
        
        if P_m < 1e-12
            continue;
        end
        
        % Posterior probabilities
        P_post = (P .* likelihoods) / P_m;
        
        % Shannon entropy
        H = -sum(P_post .* log2(P_post + 1e-12));
        
        % Expected entropy
        expected_entropy = expected_entropy + P_m * H;
    end
    
    entropy_vals(b) = expected_entropy;
end

%% 🔷 FIND OPTIMAL BETA
[~, idx_opt] = min(entropy_vals);
beta_opt = beta_real(idx_opt);

%% 🔷 PLOTTING
figure;
plot(beta_real, entropy_vals, 'b', 'LineWidth', 2); hold on; grid on;

% Mark optimal beta
plot(beta_opt, entropy_vals(idx_opt), 'ro', 'MarkerSize', 8, 'LineWidth', 2);

% Mark nulling points (+alpha and -alpha)
xline(alpha, '--k', 'LineWidth', 1.5);
xline(-alpha, '--k', 'LineWidth', 1.5);

xlabel('\beta');
ylabel('Expected Entropy');
title('BPSK: Expected Entropy vs Displacement \beta');

legend('Entropy', 'Optimal \beta', 'Nulling Points');

%% 🔷 DISPLAY RESULTS
fprintf('Optimal beta: %.4f\n', beta_opt);
fprintf('Nulling positions: +%.2f and -%.2f\n', alpha, -alpha);