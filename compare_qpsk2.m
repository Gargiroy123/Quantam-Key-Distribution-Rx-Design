clc; clear; close all;

%% 🔷 PARAMETERS
alpha2_vals = linspace(0.01, 2.5, 20); % Mean photon number (n)
M = 4;                                % QPSK
N_trials = 5000;                      % Increased for better log-scale resolution

Pe_het = zeros(size(alpha2_vals));
Pe_bayes = zeros(size(alpha2_vals));
Pe_cyclic = zeros(size(alpha2_vals));
Pe_hel = zeros(size(alpha2_vals));

%% 🔷 LOOP OVER SIGNAL ENERGY
for idx = 1:length(alpha2_vals)
    
    alpha2 = alpha2_vals(idx);
    alpha = sqrt(alpha2);
    
    % QPSK STATES (Phases: pi/4, 3pi/4, 5pi/4, 7pi/4)
    k = 0:M-1;
    theta = pi*(2*k+1)/M;
    alpha_k = alpha * exp(1i*theta);
    
    %% 🔷 1. HELSTROM LIMIT (Numerical PGM)
    % Gram matrix represents the overlap between coherent states
    G = zeros(M,M);
    for i = 1:M
        for j = 1:M
            % Overlap formula for coherent states: <alpha_i|alpha_j>
            G(i,j) = exp(-0.5*(abs(alpha_k(i))^2 + abs(alpha_k(j))^2) + ...
                         conj(alpha_k(i))*alpha_k(j));
        end
    end

    [V, D] = eig(G);
    % Ensure eigenvalues are non-negative due to precision
    sqrtD = sqrt(max(D, 0)); 
    sqrtG = V * sqrtD * V';
    Pc = (trace(sqrtG))^2 / M^2;
    Pe_hel(idx) = (1 - Pc);

    %% 🔷 2. HETERODYNE DETECTION
    errors_het = 0;
    for trial = 1:N_trials
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        % Standard AWGN for Heterodyne (Vacuum noise = 1/2 per quadrature)
        noise = (randn + 1i*randn) / sqrt(2);
        z = alpha_true + noise;
        
        [~, detected] = min(abs(z - alpha_k));
        if detected ~= true_idx, errors_het = errors_het + 1; 
        end
    end
    Pe_het(idx) = errors_het / N_trials;

    %% 🔷 3. TRUE BAYESIAN PROBING (Sequential Nulling)
    errors_bayes = 0;
    N_steps = 4; % Number of sequential measurements

    for trial = 1:N_trials
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        P = ones(1,M)/M; % Prior
        
        for step = 1:N_steps
            % Choose local oscillator to 'null' the most likely state
            [~, idx_max] = max(P);
            beta = alpha_k(idx_max);
            
            % Displaced state energy
            lambda = abs(alpha_true - beta)^2;
            m = poissrnd(lambda); 
            
            % Update using Log-Likelihood to avoid precision/factorial issues
            log_L = zeros(1,M);
            for j = 1:M
                L_j = abs(alpha_k(j) - beta)^2;
                % Log of Poisson PDF: -L + m*log(L) - log(m!)
                % Note: log(m!) is constant for all j in a single step, so we ignore it
                if L_j == 0
                    if m == 0, log_L(j) = 0; else, log_L(j) = -inf; end
                else
                    log_L(j) = -L_j + m*log(L_j);
                end
            end
            
            % Posterior update: P_new = P_old * Likelihood
            P = exp(log_L + log(P + 1e-12)); % 1e-12 prevents log(0)
            P = P / sum(P);
        end
        [~, detected] = max(P);
        if detected ~= true_idx, errors_bayes = errors_bayes + 1; end
    end
    Pe_bayes(idx) = errors_bayes / N_trials;

    %% 🔷 4. CYCLIC PROBING
    errors_cyclic = 0;
    for trial = 1:N_trials
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        P = ones(1,M)/M;
        
        for step = 1:M % Cycle through each possible state
            beta = alpha_k(step);
            lambda = abs(alpha_true - beta)^2;
            m = poissrnd(lambda);
            
            log_L = zeros(1,M);
            for j = 1:M
                L_j = abs(alpha_k(j) - beta)^2;
                if L_j == 0
                    if m == 0, log_L(j) = 0; else, log_L(j) = -inf; end
                else
                    log_L(j) = -L_j + m*log(L_j);
                end
            end
            P = exp(log_L + log(P + 1e-12));
            P = P / sum(P);
        end
        [~, detected] = max(P);
        if detected ~= true_idx, errors_cyclic = errors_cyclic + 1; end
    end
    Pe_cyclic(idx) = errors_cyclic / N_trials;
end

%% 🔷 VISUALIZATION
figure('Color', 'w', 'Position', [100 100 900 400]);

subplot(1,2,1);
plot(alpha2_vals, Pe_bayes, 'r-o', 'LineWidth', 1.5); hold on;
plot(alpha2_vals, Pe_cyclic, 'g-s', 'LineWidth', 1.5);
plot(alpha2_vals, Pe_het, 'b-d', 'LineWidth', 1.5);
plot(alpha2_vals, Pe_hel, 'k--', 'LineWidth', 2);
grid on; xlabel('Mean Photon Number |\alpha|^2'); ylabel('P_e');
title('Linear Scale');

subplot(1,2,2);
semilogy(alpha2_vals, Pe_bayes, 'r-o', 'LineWidth', 1.5); hold on;
semilogy(alpha2_vals, Pe_cyclic, 'g-s', 'LineWidth', 1.5);
semilogy(alpha2_vals, Pe_het, 'b-d', 'LineWidth', 1.5);
semilogy(alpha2_vals, Pe_hel, 'k--', 'LineWidth', 2);
grid on; xlabel('Mean Photon Number |\alpha|^2'); ylabel('P_e (Log)');
title('Log Scale');
legend('Bayesian (Sequential)','Cyclic','Heterodyne','Helstrom Limit');