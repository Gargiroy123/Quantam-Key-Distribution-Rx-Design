clc; clear; close all;

%% 🔷 PARAMETERS
M = 8;                      % QPSK
alpha2 = 0.5;               % mean photon number |alpha|^2
beta = 0.23;                % displacement
T = 1;                      % total time
dt = 0.001;                 % time resolution
t = 0:dt:T;

% True transmitted state (example: alpha3)
true_k = 3;

%% 🔷 QPSK CONSTELLATION
alpha = sqrt(alpha2) * exp(1j*(0:M-1)*2*pi/M);

% Mean photon numbers after displacement
n_k = abs(alpha - beta).^2;

%% 🔷 INITIAL POSTERIOR
P = ones(M, length(t)) / M;   % store over time
P_current = ones(M,1)/M;      % current posterior

%% 🔷 GENERATE PHOTON ARRIVALS (Poisson process)
lambda_true = n_k(true_k);

arrival_times = [];
time = 0;

while time < T
    delta = -log(rand)/lambda_true;   % exponential inter-arrival time
    time = time + delta;
    if time < T
        arrival_times = [arrival_times time];
    end
end

%% 🔷 CYCLIC PROBING SIMULATION
current_state = 1; 
event_idx = 1;

for i = 2:length(t)
    
    % Check if photon arrives
    if event_idx <= length(arrival_times) && ...
       abs(t(i) - arrival_times(event_idx)) < dt
        
        % 🔹 Photon detected → Poisson likelihood ∝ n_k
        likelihood = n_k;
        
        % Bayesian update
        P_current = P_current .* likelihood.';
        P_current = P_current / sum(P_current);
        
        % Move to next hypothesis (cyclic probing)
        current_state = mod(current_state, M) + 1;
        
        event_idx = event_idx + 1;
        
    else
        % 🔹 No photon → vacuum probability
        likelihood = exp(-n_k * dt);
        
        P_current = P_current .* likelihood.';
        P_current = P_current / sum(P_current);
    end
    
    % Store posterior
    P(:,i) = P_current;
end

%% ==========================================================
%% 🔷 EXTRACT TABLE (Posterior at fixed time points)
%% ==========================================================

t_check = 0.1:0.1:1.0;
idx = round(t_check / dt) + 1;

posterior_table = P(:, idx);

% Variable names like t_0_10
varNames = strcat('t_', strrep(compose('%.2f', t_check), '.', '_'));

T_table = array2table(posterior_table, ...
    'VariableNames', varNames, ...
    'RowNames', {'alpha1','alpha2','alpha3','alpha4'});

disp(T_table);

% Export to CSV
writetable(T_table, 'posterior_table.csv', 'WriteRowNames', true);

%% ==========================================================
%% 🔷 PLOT POSTERIOR EVOLUTION
%% ==========================================================

figure; hold on;

colors = lines(M);

% Plot posterior curves
for k = 1:M
    plot(t, P(k,:), 'LineWidth', 1.5, 'Color', colors(k,:));
end

% Mark photon arrivals
for tt = arrival_times
    xline(tt, '--k');
end

% Decision trajectory
[~, max_idx] = max(P);
stairs(t, max_idx/M, 'k', 'LineWidth', 2);

% True state reference
yline(true_k/M, '--r', 'True State');

xlabel('Temporal progress of measurement');
ylabel('a posteriori probabilities');

title('\alpha_3 case (Cyclic Bayesian Evolution)');

legend('\alpha_1','\alpha_2','\alpha_3','\alpha_4','Decision');

grid on;

%% ==========================================================
%% 🔷 OPTIONAL: SMOOTH INTERPOLATED PLOT
%% ==========================================================

t_fine = linspace(0,1,1000);
P_interp = zeros(M, length(t_fine));

for k = 1:M
    P_interp(k,:) = interp1(t, P(k,:), t_fine, 'linear');
end

figure; hold on;

for k = 1:M
    plot(t_fine, P_interp(k,:), 'LineWidth', 1.5);
end

xlabel('Temporal progress (smooth)');
ylabel('a posteriori probabilities');
title('Interpolated Posterior Evolution');

legend('\alpha_1','\alpha_2','\alpha_3','\alpha_4');

grid on;