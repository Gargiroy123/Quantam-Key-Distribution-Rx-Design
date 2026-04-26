clc; clear; close all;

%% 🔷 PARAMETERS
alpha2_vals = linspace(0.01,2.5,40);
Pe_het = zeros(size(alpha2_vals));
Pe_bayes = zeros(size(alpha2_vals));
Pe_cyclic = zeros(size(alpha2_vals));

N_trials = 10;   % increase for smooth curves
M = 4;

%% 🔷 LOOP OVER SIGNAL ENERGY
for idx = 1:length(alpha2_vals)
    
    alpha2 = alpha2_vals(idx);
    alpha = sqrt(alpha2);
    
    % QPSK STATES
    k = 0:M-1;
    theta = pi*(2*k+1)/M;
    alpha_k = alpha * exp(1i*theta);
    
    %% 🔷 HETERODYNE (nearest neighbor)
    errors_het = 0;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        % Heterodyne measurement (Gaussian noise)
        noise = (randn + 1i*randn)/sqrt(2);
        z = alpha_true + noise;
        
        % Decision: nearest constellation point
        [~, detected] = min(abs(z - alpha_k));
        
        if detected ~= true_idx
            errors_het = errors_het + 1;
        end
    end
    
    Pe_het(idx) = errors_het / N_trials;
    
    %% 🔷 BAYESIAN PROBING (QPSK)
    errors_bayes = 0;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        P = ones(1,M)/M;
        
        % 🔁 Optimize beta (2D grid)
        beta_vals = linspace(-2*alpha,2*alpha,20);
        best_beta = 0;
        min_entropy = inf;
        m_max = 6;
        
        for br = beta_vals
            for bi = beta_vals
                beta = br + 1i*bi;
                
                expected_entropy = 0;
                
                for m = 0:m_max
                    
                    likelihoods = zeros(1,M);
                    P_m = 0;
                    
                    for j = 1:M
                        lambda = abs(alpha_k(j) - beta)^2;
                        likelihoods(j) = (lambda^m / factorial(m)) * exp(-lambda);
                        P_m = P_m + P(j)*likelihoods(j);
                    end
                    
                    if P_m < 1e-10
                        continue;
                    end
                    
                    P_post = (P .* likelihoods) / P_m;
                    H = -sum(P_post .* log2(P_post + 1e-12));
                    
                    expected_entropy = expected_entropy + P_m * H;
                end
                
                if expected_entropy < min_entropy
                    min_entropy = expected_entropy;
                    best_beta = beta;
                end
            end
        end
        
        % Measurement
        lambda = abs(alpha_true - best_beta)^2;
        m = poissrnd(lambda);
        
        likelihoods = zeros(1,M);
        for j = 1:M
            lambda_j = abs(alpha_k(j) - best_beta)^2;
            likelihoods(j) = (lambda_j^m / factorial(m)) * exp(-lambda_j);
        end
        
        P_post = P .* likelihoods;
        P_post = P_post / sum(P_post);
        
        [~, detected] = max(P_post);
        
        if detected ~= true_idx
            errors_bayes = errors_bayes + 1;
        end
    end
    
    Pe_bayes(idx) = errors_bayes / N_trials;
    
    %% 🔷 CYCLIC PROBING (QPSK)
    errors_cyclic = 0;
    N_steps = 4;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        P = ones(1,M)/M;
        
        for step = 1:N_steps
            
            beta = alpha_k(mod(step-1,M)+1);  % cycle through 4 states
            
            lambda = abs(alpha_true - beta)^2;
            m = poissrnd(lambda);
            
            likelihoods = zeros(1,M);
            for j = 1:M
                lambda_j = abs(alpha_k(j) - beta)^2;
                likelihoods(j) = (lambda_j^m / factorial(m)) * exp(-lambda_j);
            end
            
            P = P .* likelihoods;
            P = P / sum(P);
        end
        
        [~, detected] = max(P);
        
        if detected ~= true_idx
            errors_cyclic = errors_cyclic + 1;
        end
    end
    
    Pe_cyclic(idx) = errors_cyclic / N_trials;
end

%% 🔷 PLOT (a)
figure;
plot(alpha2_vals, Pe_bayes, 'r', 'LineWidth', 2); hold on;
plot(alpha2_vals, Pe_cyclic, 'g', 'LineWidth', 2);
plot(alpha2_vals, Pe_het, 'b', 'LineWidth', 2);

grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability');
title('QPSK Error Probability');

legend('Bayesian','Cyclic','Heterodyne');

%% 🔷 PLOT (b) LOG SCALE
figure;
semilogy(alpha2_vals, Pe_bayes, 'r', 'LineWidth', 2); hold on;
semilogy(alpha2_vals, Pe_cyclic, 'g', 'LineWidth', 2);
semilogy(alpha2_vals, Pe_het, 'b', 'LineWidth', 2);

grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability (log)');
title('QPSK Error Probability (Log Scale)');

legend('Bayesian','Cyclic','Heterodyne');