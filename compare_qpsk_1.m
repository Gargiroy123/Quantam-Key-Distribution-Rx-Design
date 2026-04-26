clc; clear; close all;

%% 🔷 PARAMETERS
%alpha2_vals = linspace(0.01,2.5,40);
alpha2_vals = linspace(0.01,2.5,20);
M = 4;
N_trials = 3000;   % increase for smoother curves

Pe_het = zeros(size(alpha2_vals));
Pe_bayes = zeros(size(alpha2_vals));
Pe_cyclic = zeros(size(alpha2_vals));
Pe_hel = zeros(size(alpha2_vals));

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
        
        noise = (randn + 1i*randn)/sqrt(2);
        z = alpha_true + noise;
        
        [~, detected] = min(abs(z - alpha_k));
        
        if detected ~= true_idx
            errors_het = errors_het + 1;
        end
    end
    
    Pe_het(idx) = errors_het / N_trials;
    
%% 🔷 TRUE BAYESIAN PROBING 

errors_bayes = 0;
N_steps = 4;

for trial = 1:N_trials
    
    true_idx = randi(M);
    alpha_true = alpha_k(true_idx);
    
    P = ones(1,M)/M;
    
    for step = 1:N_steps
        
        % 🔥 KEY CHANGE: choose most likely hypothesis
        [~, idx_max] = max(P);
        %beta = alpha_k(idx_max);
       beta = 0.8 * alpha_k(idx_max);
        % Photon detection
        
       % lambda = abs(alpha_true - beta)^2;

       epsilon = 0.118;   % noise floor (tune this)
       lambda = abs(alpha_true - beta)^2 + epsilon;
      %  m = poissrnd(lambda);
      m = poissrnd(lambda + 0.2);
        % Bayesian update
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
        errors_bayes = errors_bayes + 1;
    end
end

Pe_bayes(idx) = errors_bayes / N_trials;
    
    %% 🔷 CYCLIC PROBING
    errors_cyclic = 0;
    N_steps = 1;
    
    for trial = 1:N_trials
        
        true_idx = randi(M);
        alpha_true = alpha_k(true_idx);
        
        P = ones(1,M)/M;
        
        for step = 1:N_steps
            
            beta = alpha_k(mod(step-1,M)+1);
            
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

%% 🔷 PLOT (LINEAR)
figure;
plot(alpha2_vals, Pe_bayes, 'r'); hold on;
plot(alpha2_vals, Pe_cyclic, 'g');
plot(alpha2_vals, Pe_het, 'b');
plot(alpha2_vals, Pe_hel, 'k');

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

grid on;
xlabel('|\alpha|^2');
ylabel('Error Probability (log)');
title('QPSK Error Probability (Log Scale)');

legend('Bayesian','Cyclic','Heterodyne','Helstrom (PGM)');