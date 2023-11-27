
warning off
clear
clc
close all
clear memory;
addpath('test_data')
addpath('fea')

dataset = {'leukemia.mat'} %%  datasets tested on the paper


collect_result_ACC=[];
collect_result_NMI=[];
collect_time=[];
for ii = 1:length(dataset)
    ii
    load(dataset{ii})
    %% hyper paramters setting
    in_X=X';
    fea=in_X;  
    
    alpha=5;lambda1=1;lambda2=2;lambda3=1;

    true_labs=index';
    gnd=true_labs(:);
    selected_class =length(unique(gnd));
    fea = double(fea);   
    nnClass =length(unique(gnd));     % The number of classes
    select_sample = [];
    select_gnd    = [];
    for i = 1:selected_class  
        idx = find(gnd == i);
        idx_sample    = fea(idx,:);
        select_sample = [select_sample;idx_sample];
        select_gnd    = [select_gnd;gnd(idx)];
    end
    rho=2.2; 
    DEBUG=0;
    fea = select_sample';
    fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
    gnd = select_gnd;
    c   = selected_class;
    X = fea;  %% genenum*cellnum
    clear fea select_gnd select_sample idx
    %
    tic()
    % ---------- initilization Z -------- %
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 10;
         options.WeightMode = 'Binary';      % Binary  HeatKernel
    options.WeightMode = 'Cosine';
    W = constructW(X',options);
    
    clear LZ DZ Z fea
    max_iter=200;
    Ctg = inv((X')*X+2*eye(size(X,2)));
    [ng, nc]=size(X);%% ng--number of genes; nc--number of cells
    kerNS=zeros(248,6);%
    % ---------- obtain similarity matrix ------- %
    [Z, E, J ,xita] =   JLSLPS(X, W, alpha,lambda1,lambda2,lambda3, rho, DEBUG,kerNS);
    clear W_ini H_ini Ctg
  
    similarity=(abs(Z)+abs(Z'))/2;
    [result_label, kerNS]= SpectralClustering(similarity, nnClass);  
    time=toc();
    % ---------- evaluation ------- %
    NMI=Cal_NMI_newused(gnd, result_label);
    ACC=ACC_ClusteringMeasure(gnd, result_label);
    fprintf(['NMI_for_ ' dataset{ii} ' is %f\n'],NMI)
    fprintf(['ACC_for_ ' dataset{ii} ' is %f\n'],ACC)
    %nz=norm(Z,1)
    %nq=norm(kerNS,1)
    collect_result_ACC(ii)=ACC;
    collect_result_NMI(ii)=NMI;
    collect_time(ii)=time;
end
% save collect_result_ACC.mat collect_result_ACC
% save collect_result_NMI.mat collect_result_NMI

%xlswrite(['NMI_collect_result_9data5' '.xlsx'],collect_result_NMI)
%xlswrite(['ACC_collect_result_9data5' '.xlsx'],collect_result_ACC)
%xlswrite(['time_collect_result_9data5' '.xlsx'],collect_time)














function [localX,coverage] = localize( C )
%C is the coefficient matrix
%[tmp,ind] = sort(C,1,'descend');
[m,n]=size(C);
localX=C;
coverage=zeros(1,n);
for i=1:n
    thr=C(i,i)/2; 
    localX(localX(:,i)<thr,i)=0;  
    coverage(1,i)=mean(C(i,i)./localX(localX(:,i)>thr,i));
end
end



function [groups, kerNS] = SpectralClustering(CKSym,n)
%--------------------------------------------------------------------------
% This function takes an adjacency matrix of a graph and computes the
% clustering of the nodes using the spectral clustering algorithm of
% Ng, Jordan and Weiss.
% CMat: NxN adjacency matrix
% n: number of groups for clustering
% groups: N-dimensional vector containing the memberships of the N points
% to the n groups obtained by spectral clustering
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
% Modified @ Chong You, 2015
%--------------------------------------------------------------------------
warning off;
N = size(CKSym,1);
% Normalized spectral clustering according to Ng & Jordan & Weiss
% using Normalized Symmetric Laplacian L = I - D^{-1/2} W D^{-1/2}
DN = diag(1./sqrt(sum(CKSym)+eps) );   %eps=2.2204e-16
LapN = speye(N) - DN * CKSym * DN;  
[~,~,vN] = svd(LapN);
kerN = vN(:,N-n+1:N);
q=kerN;
%kerN = vN(:,N-12:N);
normN = sum(kerN .^2, 2) .^.5;%% normalize the matrix U by L2-Norm
kerNS = bsxfun(@rdivide, kerN, normN + eps);  
Q=kerNS;
%-------------
%Y = pdist(kerNS);
%Z = linkage(Y);
%groups = cluster(Z,'maxclust',n);
MAXiter = 1000; % Maximum number of iterations for KMeans
REPlic = 100; % Number of replications for KMeans
groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');  
end