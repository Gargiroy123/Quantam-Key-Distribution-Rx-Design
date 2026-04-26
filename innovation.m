clc; clear; close all;

%% 🔷 PARAMETERS
alpha2_vals = linspace(0.1, 4, 15); 
M = 4;
N_trials = 5000;
N_steps = 10;        % Number of recursive "filter" steps
Q_drift = 0.01;      % INNOVATION: Process noise (phase uncertainty)

Pe_het = zeros(size(alpha2_vals));
Pe_qkf = zeros(size(alpha2_vals));
Pe_hel = zeros(size(alpha2_vals));

%% 🔷 LOOP OVER SIGNAL ENERGY
for idx = 1:length(alpha2_vals)
    alpha2 = alpha2_vals(idx);
    alpha = sqrt(alpha2);
    
    % QPSK STATES
    k = 0:M-1;
    theta = pi*(2*k+1)/M;
    alpha_k = alpha * exp(1i*theta);
    
    %% 🔷 1. HELSTROM (PGM)
    G = zeros(M,M);
    for i = 1:M
        for j = 1:M
            G(i,j) = exp(-0.5*(abs(alpha_k(i))^2 + abs(alpha_k(j))^2) + conj(alpha_k(i))*alpha_k(j));
        end
    end
    [~,D] = eig(G/M);
    Pe_hel(idx) = 1 - (sum(sqrt(max(diag(D),0))))^2;

    %% 🔷 2. HETERODYNE
    errors_het = 0;
    for trial = 1:N_trials
        true_idx = randi(M);
        z = alpha_k(true_idx) + (randn + 1i*randn)/sqrt(2);
        [~, det] = min(abs(z - alpha_k));
        if det ~= true_idx, errors_het = errors_het + 1; end
    end
    Pe_het(idx) = errors_het / N_trials;
    
    %% 🔷 3. INNOVATIVE QUANTUM KALMAN FILTER (QKF)
    errors_qkf = 0;
    alpha_step = sqrt(alpha2 / N_steps); % Correct energy partitioning
    
    for trial = 1:N_trials
        true_idx = randi(M);
        % Initial State Estimate (Belief Vector)
        P = ones(1,M)/M; 
        
        for step = 1:N_steps
            % --- PREDICT STEP ---
            % We account for phase drift by "diffusing" the probability.
            % This prevents the filter from getting stuck too early.
            P = P * (1 - Q_drift) + (ones(1,M)/M) * Q_drift; 
            
            % --- CONTROL/MEASUREMENT ---
            [~, idx_max] = max(P);
            s = 1.15; % Optimized displacement (Heuristic "Gain")
            beta = s * alpha_step * exp(1i * theta(idx_max));
            
            % Incoming photon slice
            lambda = abs(alpha_step * exp(1i * theta(true_idx)) - beta)^2;
            m = poissrnd(lambda);
            
            % --- UPDATE (CORRECTION) ---
            log_L = zeros(1,M);
            for j = 1:M
                Lj = abs(alpha_step * exp(1i * theta(j)) - beta)^2;
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
        if final_det ~= true_idx, errors_qkf = errors_qkf + 1; end
    end
    Pe_qkf(idx) = errors_qkf / N_trials;
end

%% 🔷 PLOTTING
figure('Color', 'w');
semilogy(alpha2_vals, Pe_het, 'b--d', 'LineWidth', 1.2); hold on;
semilogy(alpha2_vals, Pe_qkf, 'm-s', 'LineWidth', 2);
semilogy(alpha2_vals, Pe_hel, 'k-', 'LineWidth', 2);

grid on;
xlabel('Mean Photon Number |\alpha|^2');
ylabel('Error Probability (P_e)');
title('Innovative Quantum Kalman Filter for QPSK');
legend('Heterodyne (SQL)', 'Quantum Kalman Filter', 'Helstrom Bound');