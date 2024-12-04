function [Xtrain, Ytrain]    = construct_data(X, Y, index_tr)
%obtain a subset of the original data set and labels
%X:nxD, n--number of samples; D--dimension of data

Xtrain=X(index_tr,:);
Ytrain=Y(index_tr,:);