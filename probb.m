clc; clear; close all;

%% 🔷 PARAMETERS
%alpha2_vals = linspace(0.01,2.5,40);
alpha2_vals = linspace(0.01,2.5,20);
M = 4;
N_trials = 3000;   % increase for smoother curves
Q_drift = 0.01;

Pe_bayes = zeros(size(alpha2_vals));

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
% G = zeros(M,M);
%
%     for j = 1:M
%         G(i,j) = exp(-0.5*abs(alpha_k(i)-alpha_k(j))^2 + ...
%                      1i*imag(conj(alpha_k(i))*alpha_k(j)));
%     end
% end
% 
% % Eigen decomposition
% [V,D] = eig(G);
% sqrtG = V * sqrt(D) * V';
% 
% % Success probability (PGM)
% Pc = (trace(sqrtG))^2 / M^2;
% 
% Pe_hel(idx) = 1 - Pc;
%     %% 🔷 HETERODYNE DETECTION
%     errors_het = 0;
%     
%     for trial = 1:N_trials
%         
%         true_idx = randi(M);
%         alpha_true = alpha_k(true_idx);
%         
%         noise = sqrt((1+xi)/2) * (randn + 1i*randn);
%         z = alpha_true + noise;
%         
%         [~, detected] = min(abs(z - alpha_k));
%         
%         if detected ~= true_idx
%             errors_het = errors_het + 1;
%         end
%     end
%     
%     Pe_het(idx) = errors_het / N_trials;
    
%% 🔷 TRUE BAYESIAN PROBING 

errors_bayes = 0;
    %N_steps = 4;
    N_steps = 15;% Number of sequential measurements
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
end
dt = 0.001
% Define time points (0.1 spacing)
t_check = 0.1:0.1:1.0;

% Convert to indices
idx = round(t_check / dt) + 1;

% Extract posterior values
posterior_table = P(:, idx);

% Create variable names like t_0_10, t_0_20, ..., t_1_00
varNames = strcat('t_', strrep(sprintfc('%.2f', t_check), '.', '_'));

% Create table
T = array2table(posterior_table, ...
    'VariableNames', varNames, ...
    'RowNames', {'alpha1','alpha2','alpha3','alpha4'});

% Display
disp(T);
writetable(T, 'posterior_table.csv', 'WriteRowNames', true);
%     
%     %% 🔷 CYCLIC PROBING
%     errors_cyclic = 0;
% %    N_steps = 2;
%     N_steps = 2;
%     
%     for trial = 1:N_trials
%         
%         true_idx = randi(M);
%         alpha_true = alpha_k(true_idx);
%         
%         P = ones(1,M)/M;
%         
%         for step = 1:N_steps
%             
%             beta = alpha_k(mod(step-1,M)+1);
%             
%             lambda = eta * abs(alpha_true - beta)^2 + lambda_d + xi;
%             m = poissrnd(lambda);
%             
%             likelihoods = zeros(1,M);
%             for j = 1:M
%                 lambda_j = eta * abs(alpha_k(j) - beta)^2 + lambda_d + xi;
%                 likelihoods(j) = (lambda_j^m / factorial(m)) * exp(-lambda_j);
%             end
%             
%             P = P .* likelihoods;
%             P = P / sum(P);
%         end
%         
%         [~, detected] = max(P);
%         
%         if detected ~= true_idx
%             errors_cyclic = errors_cyclic + 1;
%         end
%     end
%     
%     Pe_cyclic(idx) = errors_cyclic / N_trials;
% 
%         %% 🔷 Quantam Kalman PROBING
%          errors_qkf = 0;
%     alpha_step = sqrt(alpha2 / 100); % Correct energy partitioning
%     
%     for trial = 1:N_trials
%         true_idx = randi(M);
%         % Initial State Estimate (Belief Vector)
%         P = ones(1,M)/M; 
%         
%         for step = 1:100
%             % --- PREDICT STEP ---
%             % We account for phase drift by "diffusing" the probability.
%             % This prevents the filter from getting stuck too early.
%             P = P * (1 - Q_drift) + (ones(1,M)/M) * Q_drift; 
%             
%             % --- CONTROL/MEASUREMENT ---
%             [~, idx_max] = max(P);
%             s = 1.5; % Optimized displacement (Heuristic "Gain")
%             beta = s * alpha_step * exp(1i * theta(idx_max));
%             
%             % Incoming photon slice
%             lambda = eta * abs(alpha_step * exp(1i * theta(true_idx)) - beta)^2 + lambda_d + xi;
%             m = poissrnd(lambda);
%             
%             % --- UPDATE (CORRECTION) ---
%             log_L = zeros(1,M);
%             for j = 1:M
%                 Lj = eta * abs(alpha_step * exp(1i * theta(j)) - beta)^2 + lambda_d + xi;
%                 % Log-Poisson likelihood
%                 if Lj == 0
%                     log_L(j) = (m == 0) * 0 + (m > 0) * -50;
%                 else
%                     log_L(j) = -Lj + m*log(Lj);
%                 end
%             end
%             
%             P = exp(log_L + log(P + 1e-12));
%             P = P / sum(P);
%         end
%         [~, final_det] = max(P);
%         if final_det ~= true_idx, errors_qkf = errors_qkf + 1; 
%         end
%     end
%     Pe_qkf(idx) = errors_qkf / N_trials;
% 
% 
% 
% 
%     %% 🔷 BONDURANT RECEIVER (TRUE SEQUENTIAL NULLING)
% errors_bondurant = 0;
% 
% for trial = 1:N_trials
%     
%     true_idx = randi(M);
%     alpha_true = alpha_k(true_idx);
%     
%     current_state = 1;
%     t = 0;
%     T_total = 1;
%     
%     while t < T_total
%         
%         % Current rate depends on current hypothesis
%         lambda = eta * abs(alpha_true - alpha_k(current_state))^2 + lambda_d + xi;
%         
%         % Draw next arrival time
%         if lambda > 0
%             delta_t = -log(rand)/lambda;
%         else
%             break;
%         end
%         
%         t = t + delta_t;
%         
%         if t >= T_total
%             break;
%         end
%         
%         % Photon detected → move to next state
%         current_state = mod(current_state, M) + 1;
%         
%     end
%     
%     detected = current_state;
%     
%     if detected ~= true_idx
%         errors_bondurant = errors_bondurant + 1;
%     end
% end
% 
% Pe_bondurant(idx) = errors_bondurant / (N_trials - 400);
% end
% 
% %% 🔷 PLOT (LINEAR)
% figure;
% plot(alpha2_vals, smoothdata(Pe_bayes), 'r', 'LineWidth',1.5); hold on;
% plot(alpha2_vals, smoothdata(Pe_cyclic), 'g', 'LineWidth',1.5);
% plot(alpha2_vals, smoothdata(Pe_het), 'b', 'LineWidth',1.5);
% plot(alpha2_vals, Pe_hel, 'k', 'LineWidth',1.5);
% plot(alpha2_vals, smoothdata(Pe_bondurant), 'c', 'LineWidth',1.5);
% plot(alpha2_vals, smoothdata(Pe_qkf), 'm--', 'LineWidth',1.5);  
% 
% grid on;
% xlabel('|\alpha|^2');
% ylabel('Error Probability');
% title('QPSK Error Probability');
% 
% legend('Bayesian','Cyclic','Heterodyne','Helstrom (PGM)','Bondurant','QKF');
% 
% %% 🔷 PLOT (LOG SCALE)
% figure;
% semilogy(alpha2_vals, smoothdata(Pe_bayes), 'r', 'LineWidth',1.5); hold on;  
% semilogy(alpha2_vals, smoothdata(Pe_cyclic), 'g', 'LineWidth',1.5);
% semilogy(alpha2_vals, smoothdata(Pe_het), 'b', 'LineWidth',1.5);
% semilogy(alpha2_vals, Pe_hel, 'k', 'LineWidth',1.5);
% semilogy(alpha2_vals, smoothdata(Pe_bondurant), 'c', 'LineWidth',1.5);
% semilogy(alpha2_vals, smoothdata(Pe_qkf), 'm--', 'LineWidth',1.5);  
% 
% grid on;
% xlabel('|\alpha|^2');
% ylabel('Error Probability (log)');
% title('QPSK Error Probability (Log Scale)');
% 
% legend('Bayesian','Cyclic','Heterodyne','Helstrom (PGM)','Bondurant','QKF');