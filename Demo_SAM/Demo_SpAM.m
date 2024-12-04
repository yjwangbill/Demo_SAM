%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo_SpAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear;clc
addpath Functions; 
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n       = 100;                      % sample size of X 
p       = 100;                      % dimension of X
sigma   = 0.5;                     % sd of noise, s/n ratio = 3
t       = 0;                        % correlation parameter of X
a       = -0.5;                     % lower bound of X
b       = 0.5;                      % upper bound of X
dG      = 1;                        % size of group 
nG      = p/dG;                     % number of groups
True_feature=[9:1:12,97:1:100];
False_feature=[1:8,13:96,101:p];      
Group   = cell(nG, 1);              % each cell contains the indices of features in a group
for i = 1:nG
    Group{i}    = ((i-1)*dG+1):(i*dG);
end

option.lambda   = 0.5:0.1:2.0;      % range of lambdas
option.maxiter  = 500;              % maximum number of iterations
option.tol      = 5e-3;             % tolerance for stopping criteria
option.criterion= 2;                % criterion for convergence: 1, maximum difference; 2, mean difference; 3, relative difference
option.verbose  = false;            % verbosely display the results
% option.verbose  = true;           % verbosely display the results
option.ktype    = 'gauss';          % Type of kernel
option.scaler   = 0.6; %0.6         % constant before the bandwidth  
NIter=1; %Number of iterations

Size_fea=zeros(NIter,1);MSE_GSAM=Size_fea;RSSE_GSAM=Size_fea;
TP=zeros(NIter,1);FP=zeros(NIter,1);
CF=zeros(NIter,1);UF=zeros(NIter,1);OF=zeros(NIter,1);

%% Run Main Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:NIter
% training data set
[Xtrain, Ytrain]    = simulate_data(n, p, sigma, a, b, t);
% validation data set
[Xvalid, Yvalid]    = simulate_data(n, p, sigma, a, b, t);
% test data set
[Xtest, Ytest]      = simulate_data(n, p, 0, a, b, t);
Xtrain = scaleData(Xtrain);
Xvalid = scaleData(Xvalid);
Xtest = scaleData(Xtest);
Strain  = getSmoother(Xtrain, Xtrain, option);
Svalid  = getSmoother(Xtrain, Xvalid, option);
Stest   = getSmoother(Xtrain, Xtest, option);
% choose best lambda by validation
[GroupSpAM_score, GroupSpAM_componentNorms, GroupSpAM_bestLambda, GroupSpAM_bestFeature]... 
= GroupSpAMselection(Ytrain, Strain, Yvalid, Svalid, Group, option);
%plotPath(option.lambda, GroupSpAM_bestLambda, GroupSpAM_componentNorms, 'GroupSpAM');

% refit with the chosen features, caculate MSE on Xtest
if ~isempty(GroupSpAM_bestFeature)
    [f0, ftrain, ftest, R]  = backfitting(Ytrain, Strain(GroupSpAM_bestFeature), Stest(GroupSpAM_bestFeature), option); 
    MSE_GroupSpAM           = mean( (sum(ftest, 2) + f0 - Ytest).^2 );
else
    MSE_GroupSpAM           = mean( (mean(Ytrain) - Ytest).^2 );
end
feature=GroupSpAM_bestFeature; 
%% Show Results
fprintf('True Feature: %s\n', int2str(True_feature));
fprintf('Selected Feature of SpAM: %s\n', int2str(feature));
Size_fea(ii)=length(feature);
[TP(ii),FP(ii),CF(ii),UF(ii),OF(ii)]=...
    Evalu_Vari_selection(feature,True_feature,False_feature);
end
size_fea=mean(Size_fea);tp=mean(TP);fp=mean(FP);
cf=sum(CF);uf=sum(UF);of=sum(OF); 
disp(['Size=',num2str(size_fea),' TP=',num2str(tp), ' FP=',num2str(fp)]);  
disp(['C=',num2str(cf), ' U=',num2str(uf), ' O=',num2str(of)]); 
 