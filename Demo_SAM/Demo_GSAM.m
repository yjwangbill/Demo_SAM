%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo_GSAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear;clc
addpath Functions;
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n       = 100;                      % sample size of X 
p       = 200;                      % dimension of X
sigma   = 0.5;                     % sd of noise, s/n ratio = 3
t       = 0;                        % correlation parameter of X
a       = -0.5;                     % lower bound of X
b       = 0.5;                      % upper bound of X
dG      = 4;                        % size of group 
nG      = p/dG;                     % number of groups
True_feature=[9:1:12,97:1:100];
False_feature=[1:8,13:96,101:p];      
partition=dG*ones(nG,1);partition2=n*ones(nG,1);
cum_part = cumsum(partition);
NIter=1;lambda=1e-1;tau=ones(nG,1);

Size_fea=zeros(NIter,1);MSE_GSAM=Size_fea;RSSE_GSAM=Size_fea;
TP=zeros(NIter,1);FP=zeros(NIter,1);
CF=zeros(NIter,1);UF=zeros(NIter,1);OF=zeros(NIter,1);

%% Run Main Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:NIter %Number of iterations

%% simulation data
% training data set
[Xtrain, Ytrain]    = simulate_data(n, p, sigma, a, b, t);
% validation data set
[Xvalid, Yvalid]    = simulate_data(n, p, sigma, a, b, t);
% test data set
[Xtest, Ytest]      = simulate_data(n, p, 0, a, b, t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Method: GSAM
options.Kernel =  'rbf' ; sigma2=1; 
options.KernelParam=sigma2;
start_ind=1;
for i=1:nG
    sel = start_ind:cum_part(i);
    K_train(:,n*(i-1)+1:n*i)=calckernel(options,Xtrain(:,sel),Xtrain(:,sel));
    K_test(:,n*(i-1)+1:n*i)=calckernel(options,Xtrain(:,sel),Xtest(:,sel));
    start_ind = cum_part(i) + 1;
end
alpha=Gene_BSR(K_train,Ytrain,lambda,partition2,tau);
ftest=K_test*alpha;
alpha1=alpha.*(abs(alpha)>=2e-1); %Threshold the group sparse vector
beta=zeros(1,nG);
for i=1:nG
    beta(i)=norm(alpha1(n*(i-1)+1:n*i));
end
temp=repmat(beta,dG,1);temp=temp(:);
feature     = find(temp > eps);feature=feature';

%% Show Results
fprintf('True Feature: %s\n', int2str(True_feature));
fprintf('Selected Feature of GSAM: %s\n', int2str(feature));
Size_fea(ii)=length(feature);
[TP(ii),FP(ii),CF(ii),UF(ii),OF(ii)]=...
    Evalu_Vari_selection(feature,True_feature,False_feature);

end
size_fea=mean(Size_fea);tp=mean(TP);fp=mean(FP);
cf=sum(CF);uf=sum(UF);of=sum(OF);
disp(['Size=',num2str(size_fea),' TP=',num2str(tp), ' FP=',num2str(fp)]);  
disp(['C=',num2str(cf), ' U=',num2str(uf), ' O=',num2str(of)]); 
 