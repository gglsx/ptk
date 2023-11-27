function [Z, E, J ,xita] =   JLSLPS(X, W, alpha,lambda1,lambda2,lambda3, rho, DEBUG,kerNS)


clear global;
global M;           %  

addpath('..\utilities\PROPACK');

if (~exist('DEBUG','var'))
    DEBUG = 0;
end

if nargin < 6
    rho = 1.9;
end

if nargin < 5
    lambda2 = 1.1;
end

if nargin < 4
    lambda3 = 1.9;
end

if nargin < 3
    lambda1 = 0.1;
end

% Construct the K-NN Graph
if nargin < 2  ||  isempty(W)
     W = constructW (X');
end
DCol = full( sum(W,2));

% unnormalized Laplacian;
D = spdiags(DCol,0,speye(size(W,1)));
L = D - W; 

normfX = norm(X,'fro');
tol1 = 1e-4;              % threshold for the error in constraint
tol2 = 1e-5;              %  threshold for the change in the solutions
[d n] = size(X);
opt.tol = tol2;            %  precision for computing the partial SVD
opt.p0 = ones(n,1);

maxIter = 500;
max_mu = 1e10;

norm2X = norm(X,2);
% mu = 1e2*tol2;
mu = min(d,n)*tol2;
eta = norm2X*norm2X*1.02;   

%% Initializing optimization variables
% intializing
E = sparse(d,n);
Y1 = zeros(d,n);
Y2 = zeros(n,n);   
Z = eye(n);
J = zeros(n, n);
XZ = zeros(d, n);   
%Q = kerNS;
xita=zeros(n,n);
nnClass=6;

sv = 10;
svp = sv;

%% Start main loop
convergenced = 0;
iter = 0;

if DEBUG
    disp(['initial,rank(Z)=' num2str(rank(Z))]);
end

while iter<maxIter
    iter = iter + 1;
    
    %copy E, J  and Z to compute the change in the solutions
    Ek = E;
    Zk = Z;
    Jk = J;
    
    XZ = X*Z;
    ZLT = Z* L';
    ZL = Z*L;
    
    %solving Z    
    %-----------Using PROPACK--------------%
    M =  lambda2* (ZLT + ZL);
    M = M + mu *X' *(XZ -X + E -Y1/mu); 
    M = M +mu *(Z- J+Y2/ mu); 
    M = Z - M/eta;
    
    [U, S, V] = svd((M+M')/2,'econ');
      
    S = diag(S);
    svp = length(find(S>1/(mu*eta)));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    if svp>=1
        S = S(1:svp)-1/(mu*eta);
    else
        svp = 1;
        S = 0;
    end

    A.U = U(:, 1:svp);
    A.s = S;
    A.V = V(:, 1:svp);
    
    Z = A.U*diag(A.s)*A.V';
    %Z = max(Z,0);
    similarity=(abs(Z)+abs(Z'))/2;
    XZ = X*Z;               
    [~, Q]= SpectralClustering(similarity, nnClass);
    % solving J  
    temp = Z+Y2/mu;
    for (i=1:n)
        for(j=1:n)
        cha=Q(i,:)-Q(j,:);
        fanshu=norm(cha,2);
        xita(i,j)=0.5*fanshu*fanshu;
        J(i,j)=max(temp(i,j) - (alpha*xita(i,j)+lambda1)/mu, 0) + min(temp(i,j) + (alpha*xita(i,j)+lambda1)/mu, 0);
        end
    end
 
    
    % solving E
    temp = X- XZ;
    temp = temp+Y1/mu;
    E = max(0, temp - lambda3/mu)+ min(0, temp + lambda3/mu);
    
    
    relChgZ = norm(Zk - Z,'fro')/normfX;
    relChgE = norm(E - Ek,'fro')/normfX;
    relChgJ = norm(J - Jk,'fro')/normfX;
    relChg =   max( max(relChgZ, relChgE), relChgJ);
    
    dY1 = X - XZ - E;
    recErr1 = norm(dY1,'fro')/normfX;    
    dY2 =  Z - J;
    recErr2 = norm(dY2,'fro')/normfX;
    recErr = max(recErr1, recErr2);
    
    convergenced = recErr <tol1  && relChg < tol2;
    
    if DEBUG
        if iter==1 || mod(iter,50)==0 || convergenced
            disp(['iter ' num2str(iter) ',mu=' num2str(mu) ...
                ',rank(Z)=' num2str(svp) ',relChg=' num2str(relChg)...
                ',recErr=' num2str(recErr)]);
        end
    end
    
    if convergenced
        break;
    else
        Y1 = Y1 + mu*dY1;
        Y2 = Y2 + mu*dY2;
        
        if mu*relChg < tol2
            mu = min(max_mu, mu*rho);
        end
    end
end


function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end


function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
