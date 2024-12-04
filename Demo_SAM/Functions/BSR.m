function x = BSR(D,y,lambda,partition)

x  =  group_lasso(D, y, lambda, partition, 1.0, 1.0);