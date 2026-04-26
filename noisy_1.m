clc; clear; close all;

%% 🔷 PARAMETERS
%alpha2_vals = linspace(0.01,2.5,40);
alpha2_vals = linspace(0.01,2.5,20);
M = 4;
N_trials = 3000;   % increase for smoother curves
Q_drift = 0.001;

Pe_het = zeros(size(alpha2_vals));
Pe_bayes = zeros(size(alpha2_vals));
Pe_cyclic = zeros(size(alpha2_vals));
Pe_hel = zeros(size(alpha2_vals));
Pe_bondurant = zeros(size(alpha2_vals));
%Pe_qkf = zeros(size(alpha2_vals));

%% 🔷 IMPAIRMENTS
eta = 0.95;           % detection efficiency
lambda_d = 0.05;     % dark count rate
xi = 0.01;            % excess noise

%% 🔷 LOOP OVER SIGNAL ENERGY
for idx = 1:length(alpha2_vals)
    
    alpha2 = alpha2_vals(idx);
    alpha = sqrt(alpha2);
    
    % QPSK STATES
    k = 0:M-1;
    theta = pi*(2*k+1)/M;
    alpha_k = alpha * exp(1i*theta);
    
    %% 🔷 HELSTROM (PGM APPROXIMATION)
    % Gram matrix
G = zeros(M,M);
for i = 1:M
    for j = 1:M
        G(i,j) = exp(-0.5*abs(alpha_k(i)-alpha_k(j))^2 + ...
                     1i*imag(conj(alpha_k(i))*alpha_k(j)));
    end
end

% Eigen decomposition
[V,D] = eig(G);
sqrtG = V * sqrt(D) * V';

% Success probability (PGM)
Pc = (trace(sqrtG))^2 / M^2;

Pe_hel(idx) = 1 - Pc;
    %% 🔷 HETERODYNE DETECTION
    errors_het = 0;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        noise = sqrt((1+xi)/2) * (randn + 1i*randn);
        z = alpha_true + noise;
        
        [~, detected] = min(abs(z - alpha_k));
        
        if detected ~= true_idx
            errors_het = errors_het + 1;
        end
    end
    
    Pe_het(idx) = errors_het / N_trials;
    
%% 🔷 TRUE BAYESIAN PROBING 

errors_bayes = 0;
    N_steps = 4; % Number of sequential measurements
alpha_step = sqrt(alpha2 / N_steps);
    for trial = 1:N_trials
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        P = ones(1,M)/M; % Prior
        
        for step = 1:N_steps
            % Choose local oscillator to 'null' the most likely state
            [~, idx_max] = max(P);
            s = 1.2; % This is a "heuristic" optimization factor
       beta = s * alpha_step * exp(1i * theta(idx_max));
           % beta = alpha_k(idx_max);
            
            % Displaced state energy
            lambda = eta * abs(alpha_true - beta)^2 + lambda_d + xi;
            m = poissrnd(lambda);
            % Update using Log-Likelihood to avoid precision/factorial issues
            log_L = zeros(1,M);
            for j = 1:M
                L_j = eta * abs(alpha_k(j) - beta)^2 + lambda_d + xi;
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
        if detected ~= true_idx, errors_bayes = errors_bayes + 1;
        end
    end
    Pe_bayes(idx) = errors_bayes / N_trials;

    
    %% 🔷 CYCLIC PROBING
    errors_cyclic = 0;
    N_steps = 2;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        P = ones(1,M)/M;
        
        for step = 1:N_steps
            
            beta = alpha_k(mod(step-1,M)+1);
            
            lambda = eta * abs(alpha_true - beta)^2 + lambda_d + xi;
            m = poissrnd(lambda);
            
            likelihoods = zeros(1,M);
            for j = 1:M
                lambda_j = eta * abs(alpha_k(j) - beta)^2 + lambda_d + xi;
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

        %% 🔷 Quantam Kalman PROBING
         errors_qkf = 0;
    alpha_step = sqrt(alpha2 / 100); % Correct energy partitioning
    
    for trial = 1:N_trials
        true_idx = randi(M);
        % Initial State Estimate (Belief Vector)
        P = ones(1,M)/M; 
        
        for step = 1:100
            % --- PREDICT STEP ---
            % We account for phase drift by "diffusing" the probability.
            % This prevents the filter from getting stuck too early.
            P = P * (1 - Q_drift) + (ones(1,M)/M) * Q_drift; 
            
            % --- CONTROL/MEASUREMENT ---
            [~, idx_max] = max(P);
            s = 1.5; % Optimized displacement (Heuristic "Gain")
            beta = s * alpha_step * exp(1i * theta(idx_max));
            
            % Incoming photon slice
            lambda = eta * abs(alpha_step * exp(1i * theta(true_idx)) - beta)^2 + lambda_d + xi;
            m = poissrnd(lambda);
            
            % --- UPDATE (CORRECTION) ---
            log_L = zeros(1,M);
            for j = 1:M
                Lj = eta * abs(alpha_step * exp(1i * theta(j)) - beta)^2 + lambda_d + xi;
                % Log-Poisson likelihood
                if Lj == 0
                    log_L(j) = (m == 0) * 0 + (m > 0) * -50;
                else
                    log_L(j) = -Lj + m*log(Lj);
                end
            end
            
            P = exp(log_L + log(P + 1e-12));
            P = P / sum(P);
        end
        [~, final_det] = max(P);
        if final_det ~= true_idx, errors_qkf = errors_qkf + 1; 
        end
    end
    Pe_qkf(idx) = errors_qkf / N_trials;




    %% 🔷 BONDURANT RECEIVER (TRUE SEQUENTIAL NULLING)

errors_bondurant = 0;

for trial = 1:N_trials
    
    true_idx = randi(M);
    alpha_true = alpha_k(true_idx);
    
    N_detect = 0;  % total photon detections
    current_state = 1;
    
    % simulate full measurement interval
    % assume Poisson process over T=1
    
    lambda = eta * abs(alpha_true - alpha_k(current_state))^2 + lambda_d + xi;
    
    % total photon count over interval
    m = poissrnd(lambda);
    
    % sequential probing effect:
    % each detection shifts state
    final_state = mod((current_state - 1 + m), M) + 1;
    
    detected = final_state;
    
    if detected ~= true_idx
        errors_bondurant = errors_bondurant + 1;
    end
end

Pe_bondurant(idx) = errors_bondurant / N_trials;
end

%% 🔷 PLOT (LINEAR)
figure;
plot(alpha2_vals, Pe_bayes, 'r'); hold on;
plot(alpha2_vals, Pe_cyclic, 'g');
plot(alpha2_vals, Pe_het, 'b');
plot(alpha2_vals, Pe_hel, 'k');
plot(alpha2_vals,Pe_bondurant,'c');

grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability');
title('QPSK Error Probability');

legend('Bayesian','Cyclic','Heterodyne','Helstrom (PGM)');

%% 🔷 PLOT (LOG SCALE)
figure;
semilogy(alpha2_vals, Pe_bayes, 'r'); hold on;  
semilogy(alpha2_vals, Pe_cyclic, 'g');
semilogy(alpha2_vals, Pe_het, 'b');
semilogy(alpha2_vals, Pe_hel, 'k');
%semilogy(alpha2_vals,Pe_qkf,'c');

grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability (log)');
title('QPSK Error Probability (Log Scale)');

legend('Bayesian','Cyclic','Heterodyne','Helstrom (PGM)');