%% some code 15/05-2019
eta_old = eta_samples
eta_k = eta_new

eta_old = eta_map
delta_old = delta_map

for samples
    while
        eta_k ~ norm(0,1)
        delta_k ~ norm(0,1)
        %Likelihood
        %exp_h ~ exp(eta_old)
        %exp_d ~ exp(delta_old)
        accept_h = min(1,g(eta_k) / g(eta_old))
        accept_d = min(1,g(delta_k) / g(delta_old))
        U_h = randn(0,1)
        U_d = randn(0,1)
        if U_h < accept_h
            eta_old = eta_k
        end
    end
end

% some code - END

%% some code 15/05-2019

likelihood = 1/(sqrt(2*pi)sigma_N)^(K*L)*exp( -1/2*((norm(X-gd*gh,'fro')^2)/(sigma_N^2)))
p_h = 1/(2*pi)^(M*L/2)*exp( -1/2*eta'*eta)
p_d = 1/(2*pi)^(K*M/2)*exp( -1/2*delta'*delta)

gMC_h = 1/((sqrt(2*pi)*sigma_N)^(K*L)*(sqrt(2*pi))^(M*L))*exp( -1/2*((norm(X-gd*gh,'fro')^2)/(sigma_N^2))+eta'*eta)
gMC_d = 1/((sqrt(2*pi)*sigma_N)^(K*L)*(sqrt(2*pi))^(K*M))*exp( -1/2*((norm(X-gd*gh,'fro')^2)/(sigma_N^2))+delta'*delta)

% some code - END

%% some code 15/05-2019

for 1:samples
    while ~accept
        eta(1) = eta_map
        g_old = g(eta(1))
        eta_new = eta(k-1) + randn(0,1)
        g_new = g(eta_new)
        accept = min(1, g(x)/g(y))
    end
end

% some code - END

