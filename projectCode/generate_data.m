% Copyright 2018 Technical University of Denmark
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
%     http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
% Author: Tommy Sonne Alstrøm <tsal@dtu.dk>
% $Date: 2018/06/11 $
% $Change: 14615 $
%
% generate simulated raman data
%
rng('default')
%rng(1337)
%rng(42)
rng(37) %test: [10 10],100,1e-2,1,2,1,5e-1,.5,3,.5,[30,80]
mapsize = [10 10]; % size of raman map [100 100] %[10 10]
Nw = 100;           % number of wavenumbers (1000) % 100
p_outlier = 1e-2;    % probability of a recording being an outlier (1e-3) % 1e-3
Nhotspots = 1;       % number of hotspots % 1
K = 2;               % number of voigts (4)
fplot = 1;           % demo plots
sig = 5e-1;          % measurement noise (1e-2)
background = .5;      % create background (1)
sbr        = 3;     % signal to background ratio (10)
hotspot_size = .5;    % hotspot size (5)
voigt_loc    = [30,80];   % vector of voigtlocs

% look for opt
if( exist('opt','var') && isfield(opt,'gendata') )
    for field = fields(opt.gendata)'
        assignin('base',char(field),opt.gendata.(char(field)));
    end
end
% set defaults

N = prod(mapsize);   % number of measurements 
wn = 1:Nw;           % wavenumber vector
DD = zeros(N,Nw);
LL = zeros(N,Nw);

gendata.sig = sig;
%% generate outlier signals with prob p_outlier
%
if( p_outlier > 0 )
    NL = binornd(N,p_outlier);

    eta = randn(NL,Nw);
    L = zeros(size(eta));
    L(:,1) = rand(NL,1);
    c = 0;
    phi = 1;
    for w=2:Nw
        L(:,w) = c + phi.*L(:,w-1) + eta(:,w);
    end

    L = L-repmat(min(L,[],2),1,Nw);
    L = L./repmat(max(L,[],2),1,Nw);

    l = exprnd(10e-2,NL,1);
    L = L.*repmat(l,1,Nw);

    if fplot
        plot(1:Nw,L);
        pause
    end
    % create outlier contribution in matrix form
    LL = zeros(N,Nw);
    inx = randperm(N,NL);
    LL(inx,:) = L;
    % store variables
    gendata.N_outlier = NL;
    gendata.lambda = l;
    gendata.Lambda = L;
end

%% generate Background
if( background )
    eta = randn(Nw,1);
    B = zeros(size(eta));
    B(1) = rand(1);
    c = 0.2;
    phi = 0.995;
    for w=2:Nw
        B(w) = c + phi*B(w-1) + eta(w);
    end
    B = smooth(B,150,'sgolay');
    if fplot
        plot(1:Nw,B)
        pause
    end
    if( length(B)>1 )
        B = B-min(B);
        B = B./max(B);
        B = B+rand;
    end

    gendata.B = B;
    % create B contribution in matrix form
    B = reshape(B,1,[]);
    B = repmat(B,N,1);

    b = betarnd(100,100,N,1);
    gendata.b = b;
    b = repmat(b,1,Nw);
    BB = b.*B;

    if fplot
        imagesc(B)
        pause
    end
else
    gendata.B = zeros(1,Nw);
    gendata.b = zeros(1,N);
    BB = zeros(N,Nw);
end

%% generate hotspots (alphas)
% generate hotspot signature
if( Nhotspots > 0 )
    mu = repmat(mapsize,Nhotspots,1) .* rand(Nhotspots,2);
    r = hotspot_size*rand(Nhotspots,1)+2;
    A = rand(Nhotspots,1);
    X = 1:mapsize(1);
    Y = 1:mapsize(2);
    [xx,yy] = meshgrid(X, Y);
    P = [reshape(xx,[],1), reshape(yy,[],1)];

    D = zeros(N,1);
    for h=1:Nhotspots
        D = D + A(h) * exp(-sum((repmat(mu(h,:),N,1)-P).^2,2)./r(h)^2);
    end
    if fplot
        mesh(X,Y,reshape(D,mapsize)')
        pause
    end

    % generate voigt parms
    mina = sbr/2;
    a = mina+mina*rand(K,1);
    w = gamrnd(21,0.5,K,1);
    if isempty(voigt_loc)
        c = w+(Nw-2*w).*rand(K,1);
    else
        c = voigt_loc';
    end
    eta = betarnd(1,1,K,1);
    theta = [c, w, eta];
    vp = voigt(theta, wn);

    if fplot
        plot(wn,vp)
        pause
    end

    spec = sum(vp,1)';
    if fplot
        plot(wn,spec)
        pause
    end

    gendata.A = repmat(a,1,N)'.*repmat(D,1,K);
    gendata.theta = theta;

    DD = gendata.A*vp;
end

%% generate noise
eta = sig^2.*randn([N,Nw]);
gendata.eta = eta;

%% generate simulated measurement matrix
gendata.DD = DD;
gendata.BB = BB;
gendata.LL = LL;
gendata.eta = eta;
X = DD + BB + LL + eta;

if fplot
    imagesc(X)
end

saving = input('wanna save?');
if saving
    save('datTest.mat','X','a','vp','DD','D');
end
shg
