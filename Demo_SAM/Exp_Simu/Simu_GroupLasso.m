%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Demo_GroupLasso
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear;clc
addpath Data Functions;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameters
n       = 100;                      % sample size of X 
p       = 200;                      % dimension of X
sigma   = 0.5;                     % sd of noise, s/n ratio = 3
t       = 0;                        % correlation parameter of X
a       = -0.5;                     % lower bound of X
b       = 0.5;                      % upper bound of X
dG      = 4;                        % size of group 
nG      = p/dG;                     % number of groups
%For Example 1 in the function simulate_data
True_feature=[9:12,97:100];
False_feature=[1:8,13:96,101:p]; 
%For Example 2 in the function simulate_data
% True_feature=1:2;
% False_feature=3:p;     
partition=dG*ones(nG,1);partition2=n*ones(nG,1);
cum_part = cumsum(partition);
NIter=100;lambda=1;
tau=ones(nG,1);

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
% Xtrain=NormalizeFea(Xtrain);
Xtrain = scaleData(Xtrain);
Xvalid = scaleData(Xvalid);
Xtest = scaleData(Xtest);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Method: GroupLasso
alpha=BSR(Xtrain,Ytrain,lambda,partition);
ftest=Xtest*alpha;

feature = find(abs(alpha)>=0.5);
feature=feature';
%% Show Results
% fprintf('True Feature: %s\n', int2str(True_feature));
% fprintf('Selected Feature of GroupLasso: %s\n', int2str(feature));
Size_fea(ii)=length(feature);
[TP(ii),FP(ii),CF(ii),UF(ii),OF(ii)]=...
    Evalu_Vari_selection(feature,True_feature,False_feature);

end
size_fea=mean(Size_fea);tp=mean(TP);fp=mean(FP);
cf=sum(CF);uf=sum(UF);of=sum(OF);
disp(['Size=',num2str(size_fea),' TP=',num2str(tp), ' FP=',num2str(fp)]);  
disp(['C=',num2str(cf), ' U=',num2str(uf), ' O=',num2str(of)]); 