%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo_Lasso
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear;clc
addpath Data Functions;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameters
True_feature=[9:1:12,97:1:100];
n       = 100;                      % sample size of X 
p       = 400;                      % dimension of X
sigma   = .5;                     % sd of noise, s/n ratio = 3
t       = 1;                        % correlation parameter of X
a       = -0.5;                     % lower bound of X
b       = 0.5;                      % upper bound of X
dG      = 4;                        % size of group 
nG      = p/dG;                     % number of groups
False_feature=[1:8, 13:96,101:p];      
partition=dG*ones(nG,1);partition2=n*ones(nG,1);
cum_part = cumsum(partition);
NIter=1;
lambda=1;

Size_fea=zeros(NIter,1);MSE_GSAM=Size_fea;RSSE_GSAM=Size_fea;
TP=zeros(NIter,1);FP=zeros(NIter,1);
CF=zeros(NIter,1);UF=zeros(NIter,1);OF=zeros(NIter,1);

for ii=1:NIter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulation data
% training data set
[Xtrain, Ytrain]    = simulate_data(n, p, sigma, a, b, t);
% validation data set
[Xvalid, Yvalid]    = simulate_data(n, p, sigma, a, b, t);
% test data set
[Xtest, Ytest]      = simulate_data(n, p, 0, a, b, t);
Xtrain = scaleData(Xtrain);
Xvalid = scaleData(Xvalid);
Xtest = scaleData(Xtest);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Method 3: Lasso
alpha=feature_sign(Xtrain,Ytrain,lambda);
ftest=Xtest*alpha;
feature = find(abs(alpha)>=1);
feature=feature';
%% Show Results
fprintf('True Feature: %s\n', int2str(True_feature));
fprintf('Selected Feature of Lasso: %s\n', int2str(feature));
Size_fea(ii)=length(feature);
[TP(ii),FP(ii),CF(ii),UF(ii),OF(ii)]=...
    Evalu_Vari_selection(feature,True_feature,False_feature);

end
size_fea=mean(Size_fea);tp=mean(TP);fp=mean(FP);
cf=sum(CF);uf=sum(UF);of=sum(OF);
disp(['Size=',num2str(size_fea),' TP=',num2str(tp), ' FP=',num2str(fp)]);  
disp(['C=',num2str(cf), ' U=',num2str(uf), ' O=',num2str(of)]); 