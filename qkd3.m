clc; clear; close all;

%% 🔷 PARAMETERS
M = 4;                 % QPSK
alpha = 1.5;           % amplitude
m_max = 5;             % photon count truncation

%% 🔷 QPSK STATES
k = 0:M-1;
theta = pi*(2*k+1)/M;
alpha_k = alpha * exp(1i * theta);

%% 🔷 PRIOR (Uniform)
P = ones(1,M)/M;

%% 🔷 BETA GRID (1D sweep along real axis)
beta_real = linspace(-3,3,200);
beta_imag = 0;

entropy_vals = zeros(size(beta_real));

%% 🔷 ENTROPY COMPUTATION LOOP
for b = 1:length(beta_real)
    
    beta = beta_real(b) + 1i*beta_imag;
    expected_entropy = 0;
    
    for m = 0:m_max
        
        P_m = 0;
        likelihoods = zeros(1,M);
        
        % Compute likelihoods
        for j = 1:M
            lambda = abs(alpha_k(j) - beta)^2;
            likelihoods(j) = (lambda^m / factorial(m)) * exp(-lambda);
            P_m = P_m + P(j) * likelihoods(j);
        end
        
        if P_m < 1e-10
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
plot(beta_real, entropy_vals, 'b', 'LineWidth', 2); hold on;
grid on;

% Mark optimal beta
plot(beta_opt, entropy_vals(idx_opt), 'ro', 'MarkerSize', 8, 'LineWidth', 2);

% Mark nulling points (real parts of alpha_k)
for i = 1:M
    xline(real(alpha_k(i)), '--k', 'LineWidth', 1);
end

xlabel('Re(\beta)');
ylabel('Expected Entropy');
title('Expected Entropy vs Displacement \beta');

legend('Entropy', 'Optimal \beta', 'Nulling Points');

%% 🔷 DISPLAY RESULTS
fprintf('Optimal beta (real axis): %.4f\n', beta_opt);
disp('QPSK state real parts (nulling positions):');
disp(real(alpha_k));