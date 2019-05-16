%% vector = matrix(:), reverse as matrix = vec2mat(vector,rows)'

%% read data
clear;
test = input('testing? true/false: ');
test = mod(test,2);

if test
    % read test data
    X = load('datTest.mat');
    X = X.X;
   
    beta_d = 0.05;
    beta_h = 1.2;
    varN = 5e-1;
    ep = 1e-12;
    lambda = 1.3;
    lambda_h = lambda;
    maxIter = 1000;
    M = 2;
else
    % read true data
    X = load('datYMikkel.mat');
    TX = load('datTXMikkel.mat');
    X = X.Y;
    TX = TX.TX;
   
    beta_d = 0.0002;
    beta_h = 0.9;
    varN = 25;
    ep = 1e-12;
    lambda = 1.5;
    lambda_h = 2;
    maxIter = 2000;    
    M = 2;
end
%Xpic = imagesc(X);

%% Initialize parameters, by choice
% rng('default');rng(42);

fprintf('Initialization completed\n\n')    
%% calculate
[K,L] = size(X);

% step (ii)
% Using a gaussian radial basis function, and matrices to avoid for loops
K1 = repmat(shiftdim((1:K*M),1),[1,K*M]);
K2 = repmat(shiftdim((1:K*M),0),[K*M,1]);
Cd = chol(exp(-beta_d*(K1-K2).^2)+ep*eye(K*M))';
fprintf('First Cholesky matrix completed\n\n')

L1 = repmat(shiftdim((1:L*M),1),[1,L*M]);
L2 = repmat(shiftdim((1:L*M),0),[L*M,1]);
Ch = chol(exp(-beta_h*(L1-L2).^2)+ep*eye(L*M))';
fprintf('Cholesky matrices completed\n\n')

% step (iii) optimize
delta = randn(K*M,1)/1e3; deltaPrior = delta;
eta = randn(L*M,1)/1e3; etaPrior = eta;
stepsizeD = 1;
stepsizeH = 1;
iter = 1;

costValue = negLogLik(delta,eta,lambda,varN,X,Cd,Ch,K,M);
w = waitbar(iter/maxIter,'Starting while-loop');

while iter<=maxIter %code for tolerance/convergence
    gd = gradL(delta,eta,lambda,varN,X,Cd,Ch,K,M,true);
    
    [delta, costValue, stepsizeD] = lineSearch(...
        @negLogLik,costValue,stepsizeD,-gd,...
        delta,eta,lambda,varN,X,Cd,Ch,K,M,true);
    
    ge = gradL(delta,eta,lambda,varN,X,Cd,Ch,K,M,false);
    
    [eta, costValue, stepsizeH] = lineSearch(...
        @negLogLik,costValue,stepsizeH,-ge,...
        delta,eta,lambda,varN,X,Cd,Ch,K,M,false);

    waitbar(iter/maxIter,w,'Optimizing')    
    iter = iter+1;
end
close(w)

%step (iv)
gD = relfun(delta,lambda,Cd,K);
gH = relfun(eta,lambda,Ch,M);

run plotting.m

%% change of variables (eq 17)
function gMat = relfun(optvar,lambda,cov,matRows)
% from equation 17
gMat = vec2mat(fInv(cov'*optvar,lambda,cov),matRows)';
end

%% negative log likelighood/ cost function (eq 22) (step i)
function L = negLogLik(delta,eta,lambda,varN,X,Cd,Ch,K,M)

L = 0.5*(1/varN*norm(X-relfun(delta,lambda,Cd,K)*...
    relfun(eta,lambda,Ch,M),'fro')^2+delta'*delta + eta'*eta);
end

%% gradient neg log lik (eq 23)
function gL = gradL(delta,eta,lambda,varN,X,Cd,Ch,K,M,optDelta)
gd = relfun(delta,lambda,Cd,K);
gh = relfun(eta,lambda,Ch,M);
if optDelta
    % gradient irt delta
    tmpVec = (gd*gh-X)*gh';
    gL = 1/varN*((tmpVec(:).*gradFInv(Cd'*delta,lambda,Cd))'*Cd')'+delta;
else
    % gradient irt eta
    tmpVec = gd'*(gd*gh-X);
    gL = 1/varN*((tmpVec(:).*gradFInv(Ch'*eta,lambda,Ch))'*Ch)'+eta;
end
end

%% inverse link function (25)
function MatVec = fInv(vec,lambda,cov)
eps = 1e-10;
% erf defined on [-1,1], so tmp [0,1]
if nargin >= 3
    tmp = eps+0.5-0.5*erf(vec./(sqrt(2*diag(cov))));
    MatVec = max(-lambda^-1*log(tmp),0);
%     MatVec(isnan(MatVec)) = 0;
else
    tmp = eps+0.5-0.5*erf(vec/(sqrt(2)));
    MatVec = max(-lambda^-1*log(tmp),0);
%     MatVec(isnan(MatVec)) = 0;
end
end

%% derivative of inv link function (26)
function gradVec = gradFInv(vec,lambda,cov)
    if nargin >= 3
        gradVec = (sqrt(2*pi*diag(cov))*lambda).^-1.*...
            exp(lambda*fInv(vec,lambda,cov)-vec.^2./(2*diag(cov)));
    else
        gradVec = (sqrt(2*pi)*lambda).^-1*exp(lambda*fInv(vec,lambda)-vec.^2/(sqrt(2)));
    end
end

%% Line search
function [optPar, costValue, stepsize] = lineSearch(...
    costFun,costValue,stepsize,negGrad,delta,eta,lambda,varN,X,Cd,Ch,K,M,optDelta)
maxK = 5;
k = 0;
if optDelta
    newCost = feval(costFun,delta + stepsize*negGrad,eta,lambda,varN,X,Cd,Ch,K,M);
    improve = (newCost <= costValue);
    if improve
        while improve && k<maxK
            optPar = delta + stepsize*negGrad;
            costValue = newCost;
            stepsize = 2*stepsize;
            newCost = feval(costFun,delta + stepsize*negGrad,eta,lambda,varN,X,Cd,Ch,K,M);
            improve = (newCost <= costValue);
            k = k + 1;
        end
        if improve
            optPar = delta + stepsize*negGrad;
            costValue = newCost;
        end
    else
        while ~improve && k<maxK
            stepsize = 0.5*stepsize;
            newCost = feval(costFun,delta + stepsize*negGrad,eta,lambda,varN,X,Cd,Ch,K,M);
            improve = (newCost <= costValue);
            k = k + 1;
        end
        if improve
            optPar = delta + stepsize*negGrad;
            costValue = newCost;
        else
            optPar = delta;
        end
    end
else
    newCost = feval(costFun,delta,eta + stepsize*negGrad,lambda,varN,X,Cd,Ch,K,M);
    improve = (newCost <= costValue);
    if improve
        while improve && k<maxK
            optPar = eta + stepsize*negGrad;
            costValue = newCost;
            stepsize = 2*stepsize;
            newCost = feval(costFun,delta,eta + stepsize*negGrad,lambda,varN,X,Cd,Ch,K,M);
            improve = (newCost <= costValue);
            k = k + 1;
        end
        if improve
            optPar = eta + stepsize*negGrad;
            costValue = newCost;
        end
    else
        while ~improve && k<maxK
            stepsize = 0.5*stepsize;
            newCost = feval(costFun,delta,eta + stepsize*negGrad,lambda,varN,X,Cd,Ch,K,M);
            improve = (newCost <= costValue);
            k = k + 1;
        end
        if improve
            optPar = eta + stepsize*negGrad;
            costValue = newCost;
        else
            optPar = eta;
        end
    end
end
end