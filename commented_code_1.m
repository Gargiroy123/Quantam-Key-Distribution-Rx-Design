clc; clear; close all;

%% 🔷 PARAMETERS
% Sweep over mean photon number |alpha|^2
alpha2_vals = linspace(0.01,2.5,20);

M = 4;                      % QPSK alphabet size
N_trials = 3000;            % Monte Carlo trials
%Q_drift = 0.01;             % Drift factor

% Preallocate error probabilities
Pe_het = zeros(size(alpha2_vals));
Pe_bayes = zeros(size(alpha2_vals));
Pe_cyclic = zeros(size(alpha2_vals));
Pe_hel = zeros(size(alpha2_vals));
Pe_bondurant = zeros(size(alpha2_vals));
%Pe_qkf = zeros(size(alpha2_vals));

%% 🔷 IMPAIRMENTS (Physical detection model)
eta = 0.95;        % Detection efficiency
lambda_d = 0.05;   % Dark count rate
xi = 0.01;         % Excess noise

%% 🔷 MAIN LOOP OVER SIGNAL ENERGY
for idx = 1:length(alpha2_vals)
    
    % Current mean photon number
    alpha2 = alpha2_vals(idx);
    alpha = sqrt(alpha2);
    
    %% 🔷 QPSK CONSTELLATION
    % Phase points: theta_k = π(2k+1)/M
    k = 0:M-1;
    theta = pi*(2*k+1)/M;
    
    % Coherent states: α_k = α e^{iθ_k}
    alpha_k = alpha * exp(1i*theta);
    
    %% ==========================================================
    %% 🔷 HELSTROM BOUND (PGM Approximation)
    %% ==========================================================
    
    % Gram matrix: G_ij = <α_i | α_j>
    G = zeros(M,M);
    for i = 1:M
        for j = 1:M
            G(i,j) = exp(-0.5*abs(alpha_k(i)-alpha_k(j))^2 + ...
                         1i*imag(conj(alpha_k(i))*alpha_k(j)));
        end
    end
    
    % Eigen decomposition: G = VDV†
    [V,D] = eig(G);
    
    % Matrix square root: √G = V√DV†
    sqrtG = V * sqrt(D) * V';
    
    % Success probability (PGM)
    % Pc = (Tr(√G))^2 / M^2
    Pc = (trace(sqrtG))^2 / M^2;
    
    % Error probability
    Pe_hel(idx) = 1 - Pc;
    
    %% ==========================================================
    %% 🔷 HETERODYNE DETECTION (Classical Receiver)
    %% ==========================================================
    
    errors_het = 0;
    
    for trial = 1:N_trials
        
        % Random transmitted state
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        % Add complex Gaussian noise (heterodyne)
        shot_noise = sqrt((1+xi)/2) * (randn + 1i*randn);
        z = alpha_true + shot_noise;
        
        % Nearest neighbour detection
        [~, detected] = min(abs(z - alpha_k));
        
        if detected ~= true_idx
            errors_het = errors_het + 1;
        end
    end
    
    Pe_het(idx) = errors_het / N_trials;
    
    %% ==========================================================
    %% 🔷 TRUE BAYESIAN PROBING (Adaptive Receiver)
    %% ==========================================================
    
    errors_bayes = 0;
    
    N_steps = 15;  % Number of sequential measurements
    
    % Energy conservation:
    % |alpha|^2 = N_steps * |alpha_step|^2
    alpha_step = sqrt(alpha2 / N_steps);
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        % Initial prior (uniform)
        P = ones(1,M)/M;
        
        for step = 1:N_steps
            
            % Select most probable hypothesis
            [~, idx_max] = max(P);
            
            % Adaptive displacement (nulling most likely state)
            s = 1.2; % heuristic scaling
            beta = s * alpha_step * exp(1i * theta(idx_max));
            
            % Photon detection model (Poisson)
            lambda = eta * abs(alpha_true - beta)^2 + lambda_d + xi;
            m = poissrnd(lambda);
            
            % Compute log-likelihoods
            log_L = zeros(1,M);
            for j = 1:M
                L_j = eta * abs(alpha_k(j) - beta)^2 + lambda_d + xi;
                
                % Log-Poisson likelihood:
                % log P = -λ + m log λ
                if L_j == 0
                    if m == 0
                        log_L(j) = 0;
                    else
                        log_L(j) = -inf;
                    end
                else
                    log_L(j) = -L_j + m*log(L_j);
                end
            end
            
            % Bayesian update:
            % P_new ∝ P_old × likelihood
            P = exp(log_L + log(P + 1e-12));
            P = P / sum(P);
        end
        
        % Final decision
        [~, detected] = max(P);
        
        if detected ~= true_idx
            errors_bayes = errors_bayes + 1;
        end
    end
    
    Pe_bayes(idx) = errors_bayes / N_trials;
    
    %% ==========================================================
    %% 🔷 CYCLIC PROBING (Non-adaptive sequence)
    %% ==========================================================
    
    errors_cyclic = 0;
    N_steps = 2;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        % Uniform prior
        P = ones(1,M)/M;
        
        for step = 1:N_steps
            
            % Fixed probing sequence
            beta = alpha_k(mod(step-1,M)+1);
            
            % Photon detection
            lambda = eta * abs(alpha_true - beta)^2 + lambda_d + xi;
            m = poissrnd(lambda);
            
            % Compute likelihoods
            likelihoods = zeros(1,M);
            for j = 1:M
                lambda_j = eta * abs(alpha_k(j) - beta)^2 + lambda_d + xi;
                likelihoods(j) = (lambda_j^m / factorial(m)) * exp(-lambda_j);
            end
            
            % Bayesian update (no adaptive control)
            P = P .* likelihoods;
            P = P / sum(P);
        end
        
        % Decision
        [~, detected] = max(P);
        
        if detected ~= true_idx
            errors_cyclic = errors_cyclic + 1;
        end
    end
    
    Pe_cyclic(idx) = errors_cyclic / N_trials;
%     
%     %% ==========================================================
%     %% 🔷 QUANTUM KALMAN FILTER (Adaptive + Drift Model)
%     %% ==========================================================
%     
%     errors_qkf = 0;
%     
%     % Divide energy into many small slices
%     alpha_step = sqrt(alpha2 / 100);
%     
%     for trial = 1:N_trials
%         
%         true_idx = randi(M);
%         
%         % Initial belief
%         P = ones(1,M)/M;
%         
%         for step = 1:100
%             
%             % --- PREDICTION STEP ---
%             % Add uncertainty (drift)
%             P = P * (1 - Q_drift) + (ones(1,M)/M) * Q_drift;
%             
%             % --- CONTROL ---
%             [~, idx_max] = max(P);
%             s = 1.5;
%             beta = s * alpha_step * exp(1i * theta(idx_max));
%             
%             % Measurement
%             lambda = eta * abs(alpha_step * exp(1i * theta(true_idx)) - beta)^2 ...
%                      + lambda_d + xi;
%             m = poissrnd(lambda);
%             
%             % --- UPDATE STEP ---
%             log_L = zeros(1,M);
%             for j = 1:M
%                 Lj = eta * abs(alpha_step * exp(1i * theta(j)) - beta)^2 ...
%                      + lambda_d + xi;
%                 
%                 if Lj == 0
%                     log_L(j) = (m == 0)*0 + (m > 0)*(-50);
%                 else
%                     log_L(j) = -Lj + m*log(Lj);
%                 end
%             end
%             
%             P = exp(log_L + log(P + 1e-12));
%             P = P / sum(P);
%         end
%         
%         [~, final_det] = max(P);
%         
%         if final_det ~= true_idx
%             errors_qkf = errors_qkf + 1;
%         end
%     end
%     
%     Pe_qkf(idx) = errors_qkf / N_trials;
%     
    %% ==========================================================
    %% 🔷 BONDURANT RECEIVER (Sequential Nulling)
    %% ==========================================================
    
    errors_bondurant = 0;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        current_state = 1;
        t = 0;
        T_total = 1;
        
        % Sequential photon arrivals (Poisson process)
        while t < T_total
            
            lambda = eta * abs(alpha_true - alpha_k(current_state))^2 ...
                     + lambda_d + xi;
            
            % Next arrival time
            if lambda > 0
                delta_t = -log(rand)/lambda;
            else
                break;
            end
            
            t = t + delta_t;
            
            if t >= T_total
                break;
            end
            
            % Move to next hypothesis
            current_state = mod(current_state, M) + 1;
        end
        
        detected = current_state;
        
        if detected ~= true_idx
            errors_bondurant = errors_bondurant + 1;
        end
    end
    
    Pe_bondurant(idx) = errors_bondurant / N_trials;

end

%% ==========================================================
%% 🔷 PLOTTING
%% ==========================================================

figure;
plot(alpha2_vals, smoothdata(Pe_bayes), 'r', 'LineWidth',1.5); hold on;
plot(alpha2_vals, smoothdata(Pe_cyclic), 'g', 'LineWidth',1.5);
plot(alpha2_vals, smoothdata(Pe_het), 'b', 'LineWidth',1.5);
plot(alpha2_vals, Pe_hel, 'k', 'LineWidth',1.5);
plot(alpha2_vals, smoothdata(Pe_bondurant), 'c', 'LineWidth',1.5);
%plot(alpha2_vals, smoothdata(Pe_qkf), 'm--', 'LineWidth',1.5);

grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability');
title('QPSK Error Probability');

legend('Bayesian','Cyclic','Heterodyne','Helstrom (PGM)','Bondurant');

figure;
semilogy(alpha2_vals, smoothdata(Pe_bayes), 'r', 'LineWidth',1.5); hold on;
semilogy(alpha2_vals, smoothdata(Pe_cyclic), 'g', 'LineWidth',1.5);
semilogy(alpha2_vals, smoothdata(Pe_het), 'b', 'LineWidth',1.5);
semilogy(alpha2_vals, Pe_hel, 'k', 'LineWidth',1.5);
semilogy(alpha2_vals, smoothdata(Pe_bondurant), 'c', 'LineWidth',1.5);
%semilogy(alpha2_vals, smoothdata(Pe_qkf), 'm--', 'LineWidth',1.5);

grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability (log)');
title('QPSK Error Probability (Log Scale)');

legend('Bayesian','Cyclic','Heterodyne','Helstrom (PGM)','Bondurant');