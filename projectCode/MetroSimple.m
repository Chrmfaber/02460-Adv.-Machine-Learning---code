

Z = gH;
Za = zeros(1,Z);


% PRIOR OVER SCALE PARAMETERS
B = 1;
% DEFINE LIKELIHOOD
likelihood = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y))','y','A','B');

% DEFINE THE POSTERIOR
p = inline('(B.^A/gamma(A)).*y.^(A-1).*exp(-(B.*y)).*sin(pi*A).^2','y','A','B');
% are we sampling a posterior, that we already know?!

% SAMPLE FROM p(A | y = 1.5)
y = 1.5;

% INITIALIZE THE METROPOLIS-HASTINGS SAMPLER
% DEFINE PROPOSAL DENSITY
q = inline('normpdf(x,mu)','x','mu');
 
% MEAN FOR PROPOSAL DENSITY
mu = gH[i]  %mu = 5;

% SOME CONSTANTS
nSamples = 5000;
% INTIIALZE SAMPLER
x = zeros(1 ,nSamples);
x(1) = mu;
t = 1;
 
% RUN METROPOLIS-HASTINGS SAMPLER
while t < nSamples
    t = t+1;
 
    % SAMPLE FROM PROPOSAL
    xStar = normrnd(mu);
 
    % CORRECTION FACTOR
    c = q(x(t-1),mu)/q(xStar,mu);
 
    % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
    alpha = min([1, p(y,xStar,B)/p(y,x(t-1),B)*c]);
 
    % ACCEPT OR REJECT?
    u = rand;
    if u < alpha
        x(t) = xStar;
    else
        x(t) = x(t-1);
    end
end