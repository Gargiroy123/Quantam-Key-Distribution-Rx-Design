clc; clear; close all;

M = 4;                      % QPSK
alpha2 = 0.5;               % mean photon number |alpha|^2
beta = 0.23;                % displacement
T = 1;                      % total time
dt = 0.001;
t = 0:dt:T;

% True state index (example: alpha3)
true_k = 3;

% Define constellation (QPSK)
alpha = sqrt(alpha2)*exp(1j*(0:M-1)*2*pi/M);

% Displaced means
n_k = abs(alpha - beta).^2;

P = ones(M, length(t)) / M;   % posterior over time
P_current = ones(M,1)/M;      % current posterior

% Generate photon arrival times (true state)
lambda_true = n_k(true_k);

arrival_times = [];
time = 0;

while time < T
    delta = -log(rand)/lambda_true;
    time = time + delta;
    if time < T
        arrival_times = [arrival_times time];
    end
end

current_state = 1; % cyclic probing starts at alpha1
event_idx = 1;

for i = 2:length(t)
    
    % Check if photon arrives at this time
    if event_idx <= length(arrival_times) && ...
       abs(t(i) - arrival_times(event_idx)) < dt
        
        % -------- Photon detection update --------
        
        likelihood = n_k; % proportional to rate
        
        P_current = P_current .* likelihood.';
        P_current = P_current / sum(P_current);
        
        % Move to next state (cyclic probing)
        current_state = mod(current_state, M) + 1;
        
        event_idx = event_idx + 1;
        
    else
        % -------- No photon update (vacuum evolution) --------
        
        likelihood = exp(-n_k*dt);
        
        P_current = P_current .* likelihood.';
        P_current = P_current / sum(P_current);
    end
    

    P(:,i) = P_current;
end

figure;

colors = lines(M);

for k = 1:M
    plot(t, P(k,:), 'LineWidth', 1.5); hold on;
end

xlabel('Temporal progress of measurement');
ylabel('a posteriori probabilities');

legend('\alpha_1','\alpha_2','\alpha_3','\alpha_4');

grid on;

for tt = arrival_times
    xline(tt, '--k');
end

[~, max_idx] = max(P);

stairs(t, max_idx/M, 'k', 'LineWidth', 2); % normalized for plotting

[~, max_idx] = max(P);

stairs(t, max_idx/M, 'k', 'LineWidth', 2); % normalized for plotting

 %
 % 
 % -------- Extract table --------
%t_check = [0.15 0.35 0.54 0.71 1.0];
% t_check = linspace(0.1, 1.0, 10); 
% idx = round(t_check / dt) + 1;
% 
% posterior_table = P(:, idx);
% 
% T = array2table(posterior_table, ...
%     'VariableNames', {'t1_0_15','t2_0_35','t3_0_54','t4_0_71','t5_1_0'}, ...
%     'RowNames', {'alpha1','alpha2','alpha3','alpha4'});
% 
% disp(T);

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

% Time points (from paper)
t_pts = [0 0.15 0.35 0.54 0.71 1];

% Posterior values (α1 case)
P_alpha1 = [ ...
    0.25   0.024 0.049 0.061 0.139 0.075;  % alpha1
    0.25   0.277 0.423 0.090 0.291 0.256;  % alpha2
    0.25   0.403 0.105 0.131 0.302 0.434;  % alpha3
    0.25   0.277 0.423 0.718 0.268 0.236]; % alpha4

% Create fine time for smooth lines
t_fine = linspace(0,1,500);

% Interpolate
P_interp = zeros(4, length(t_fine));
for k = 1:4
    P_interp(k,:) = interp1(t_pts, P_alpha1(k,:), t_fine, 'linear');
end

% Plot
figure;
hold on;

colors = lines(4);

for k = 1:4
    plot(t_fine, P_interp(k,:), 'LineWidth', 1.5, 'Color', colors(k,:));
end

% Vertical lines at detection times
for tt = t_pts(2:end-1)
    xline(tt, '--k');
end

xlabel('Temporal progress of measurement');
ylabel('a posteriori probabilities');

title('\alpha_1 case');

legend('\alpha_1','\alpha_2','\alpha_3','\alpha_4');

grid on;