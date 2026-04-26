clc; clear; close all;

%% 🔷 PARAMETERS
alpha2_vals = linspace(0.01,2.5,50);   % |alpha|^2
Pe_het = zeros(size(alpha2_vals));
Pe_hel = zeros(size(alpha2_vals));
Pe_bayes = zeros(size(alpha2_vals));
Pe_cyclic = zeros(size(alpha2_vals));
N_trials = 10;   % Monte Carlo trials

%% 🔷 LOOP OVER SIGNAL ENERGY
for idx = 1:length(alpha2_vals)
    
    alpha2 = alpha2_vals(idx);
    alpha = sqrt(alpha2);
    
    % BPSK states
    alpha_k = [alpha, -alpha];
    
    %% 🔷 Helstrom bound (analytical)
    Pe_hel(idx) = 0.5 * (1 - sqrt(1 - exp(-4*alpha2)));
    
    %% 🔷 Heterodyne detection (analytical approx)
    Pe_het(idx) = 0.5 * erfc(alpha / sqrt(2));
    
    %% 🔷 Bayesian receiver (simulation)
    errors = 0;
    
    for trial = 1:N_trials
        
        % Random transmitted state
        true_idx = randi(2);
        alpha_true = alpha_k(true_idx);
        
        % Prior
        P = [0.5, 0.5];
        
        % 🔁 Single-step optimal beta (entropy search)
        beta_grid = linspace(-2*alpha, 2*alpha, 20);
        best_beta = 0;
        min_entropy = inf;
        
        for b = 1:length(beta_grid)
            beta = beta_grid(b);
            expected_entropy = 0;
            m_max = 10;
            for m = 0:m_max
                likelihoods = zeros(1,2);
                P_m = 0;
                
                for j = 1:2
                    lambda = abs(alpha_k(j) - beta)^2;
                    likelihoods(j) = (lambda^m / factorial(m)) * exp(-lambda);
                    P_m = P_m + P(j)*likelihoods(j);
                end
                
                if P_m < 1e-10
                    continue;
                end
                
                P_post = (P .* likelihoods) / P_m;
%                 H = -sum(P_post .* log2(P_post + 1e-12));
                H = -sum(P_post .* log2(P_post + 1e-12));
                expected_entropy = expected_entropy + P_m * H;
            end
            
            if expected_entropy < min_entropy
                min_entropy = expected_entropy;
                best_beta = beta;
            end
        end
        
        %% 🔷 Measurement with optimal beta
        lambda = abs(alpha_true - best_beta)^2;
        m = poissrnd(lambda);
        
        % Bayesian update
        likelihoods = zeros(1,2);
        for j = 1:2
            lambda_j = abs(alpha_k(j) - best_beta)^2;
            likelihoods(j) = (lambda_j^m / factorial(m)) * exp(-lambda_j);
        end
        
        P_post = P .* likelihoods;
        P_post = P_post / sum(P_post);
        
        % Decision
        [~, detected] = max(P_post);
        
        if detected ~= true_idx
            errors = errors + 1;
        end
    end
    
    Pe_bayes(idx) = errors / N_trials;

    %% 🔷 CYCLIC PROBING (MULTI-STEP)

errors_cyclic = 0;
N_steps = 4;   % number of probing steps

for trial = 1:N_trials
    
    true_idx = randi(2);
    alpha_true = alpha_k(true_idx);
    
    P = [0.5, 0.5];   % prior
    
    for step = 1:N_steps
        
        % 🔁 Cycle between +alpha and -alpha
        if mod(step,2) == 1
            beta = alpha;     % +alpha
        else
            beta = -alpha;    % -alpha
        end
        
        % Photon detection
        lambda = abs(alpha_true - beta)^2;
        m = poissrnd(lambda);
        
        % Bayesian update
        likelihoods = zeros(1,2);
        for j = 1:2
            lambda_j = abs(alpha_k(j) - beta)^2;
            likelihoods(j) = (lambda_j^m / factorial(m)) * exp(-lambda_j);
        end
        
        P = P .* likelihoods;
        P = P / sum(P);
    end
    
    % Final decision
    [~, detected] = max(P);
    
    if detected ~= true_idx
        errors_cyclic = errors_cyclic + 1;
    end
end

Pe_cyclic(idx) = errors_cyclic / N_trials;
end

%% 🔷 PLOT (a) Linear scale
figure;
plot(alpha2_vals, Pe_bayes, 'r', 'LineWidth', 2); hold on;
plot(alpha2_vals, Pe_het, 'b', 'LineWidth', 2);
plot(alpha2_vals, Pe_hel, 'k', 'LineWidth', 2);
plot(alpha2_vals, Pe_cyclic, 'g', 'LineWidth', 2);
grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability');
title('(a) Error Probability vs Signal Energy');

%% 🔷 PLOT (b) Log scale
figure;
semilogy(alpha2_vals, Pe_bayes, 'r', 'LineWidth', 2); hold on;
semilogy(alpha2_vals, Pe_het, 'b', 'LineWidth', 2);
semilogy(alpha2_vals, Pe_hel, 'k', 'LineWidth', 2);
semilogy(alpha2_vals, Pe_cyclic, 'g', 'LineWidth', 2);
grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability (log scale)');
title('(b) Log-scale Error Probability');
