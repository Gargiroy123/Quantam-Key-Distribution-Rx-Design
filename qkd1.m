clc; clear;

M = 4;
alpha = 1.5;
k = 0:M-1;
theta = pi*(2*k+1)/M;
alpha_k = alpha * exp(1i * theta);

P = ones(1,M)/M;   % Prior probabilities

true_idx = randi(M);
alpha_true = alpha_k(true_idx);

N_steps = 10;   % number of probing steps

for step = 1:N_steps
    
    % 🔴 Choose displacement (Bayesian probing strategy)
    % Null the most probable hypothesis
    [~, max_idx] = max(P);
    beta = alpha_k(max_idx);
    
    % 🔵 Photon detection (Poisson)
    lambda = abs(alpha_true - beta)^2;
    m = poissrnd(lambda);
    
    % 🔁 Bayesian Update
    for j = 1:M
        lambda_j = abs(alpha_k(j) - beta)^2;
        likelihood = (lambda_j^m / factorial(m)) * exp(-lambda_j);
        P(j) = P(j) * likelihood;
    end
    
    % Normalize
    P = P / sum(P);
    
    % Display progress
    fprintf('Step %d: Probabilities = ', step);
    disp(P);
end

[~, detected_idx] = max(P);

fprintf('True state: %d\n', true_idx);
fprintf('Detected state: %d\n', detected_idx);