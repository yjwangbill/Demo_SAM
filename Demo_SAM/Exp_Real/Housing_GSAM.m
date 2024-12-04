%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Housing_GSAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear all;clc
addpath  Functions;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load data
filename = '/Users/jacky/Desktop/Data/农业数据/cows/cows_data/data.xlsx';
data = xlsread(filename,4);
X=data(:,1:43); %(NxD)
Y=data(:,44);%(Nx1)
num_all=size(X,1);
X = scaleData(X);
% n_valid=floor(n_train/2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parameters
NIter=1;lambda=1;
%% parameters
m=100;n_train=num_all-m; n=n_train;
p=size(X,2);
dG      = 1;                        % size of group 
nG      = floor(p/dG);                    % number of groups
tau=ones(nG,1); 
partition=dG*ones(nG,1);partition2=n*ones(nG,1);
cum_part = cumsum(partition); 
options.Kernel =  'rbf' ; sigma2=0.5;
options.KernelParam=sigma2;

RMSE_GSAM=zeros(NIter,1);RSSE_GSAM=zeros(NIter,1);
Feature=cell(NIter,1);
for ii=1:NIter
    
Index=randperm(num_all);
index_tt=Index(1:m);%index_valid=Index(m+1:m+n_valid);
index_tr=Index(m+1:end);
% training data set
[Xtrain, Ytrain]    = construct_data(X, Y, index_tr);
% validation data set
% [Xvalid, Yvalid]    = construct_data(X, Y, index_valid);
% test data set
[Xtest, Ytest]      = construct_data(X, Y, index_tt);

start_ind=1;
for i=1:nG
    sel = start_ind:cum_part(i);
    K_train(:,n*(i-1)+1:n*i)=calckernel(options,Xtrain(:,sel),Xtrain(:,sel));
    K_test(:,n*(i-1)+1:n*i)=calckernel(options,Xtrain(:,sel),Xtest(:,sel));
    start_ind = cum_part(i) + 1;
end

alpha=Gene_BSR(K_train,Ytrain,lambda,partition2,tau);
ftest=K_test*alpha;
RMSE_GSAM(ii)= mean( (ftest - Ytest).^2./Ytest.^2);
RSSE_GSAM(ii)=calcRSSE(Ytest,ftest);
figure;plot(alpha)
set(gca,'XTick',0:406:5278);
title('\alpha for Housing data')
alpha1=alpha.*(abs(alpha)>=10);
beta=zeros(1,nG);
for i=1:nG
    beta(i)=norm(alpha1(n*(i-1)+1:n*i));
end
% % figure;plot(beta);
temp=repmat(beta,dG,1);temp=temp(:);
% aa=sort(beta,'descend');thre=aa(3);
feature     = find(temp > eps);feature=feature';
Feature{ii}=feature;
fprintf('Selected Feature of GSAM: %s\n', int2str(feature));

end
rmse_GSAM=mean(RMSE_GSAM);rsse_GSAM=mean(RSSE_GSAM);
std_rmse=std(RMSE_GSAM);std_rsse=std(RSSE_GSAM); 
Results=[rmse_GSAM,std_rmse,rsse_GSAM,std_rsse];
 
fprintf('RMSE of GSAM: %f\n', rmse_GSAM);
fprintf('RSSE of GSAM: %f\n',rsse_GSAM);
% save GSAM_Housing_170211_thres5.mat