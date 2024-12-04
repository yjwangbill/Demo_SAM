%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author  :   Junming Yin
% Contact :   junmingy@email.arizona.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [f0, ftrain, ftest, R]    = backfitting(Ytrain, Strain, Stest, option)
    % standard backfitting algorithm for additive models
    
    maxiter = option.maxiter;
    tol     = option.tol;
    verbose = option.verbose; 
        
    n           = length(Ytrain);
    ntest       = size(Stest{1}, 1);
    p           = length(Strain);       % dimension of selected features
    
    f0          = mean(Ytrain);         % intercept
    ftrain      = zeros(n, p);          % fitted values at training data points
    ftest       = zeros(ntest, p);      % predicted values at test data points
    R           = Ytrain - f0;          % residuals
    
    epsilon     = 1;
    iter        = 0;
    while (epsilon > tol && iter < maxiter)
        ftrain_old  = ftrain;
        for j   = 1 : p
            R   = R + ftrain(:,j);         

            ftrain(:,j) = Strain{j}*R;
            ftrain(:,j) = ftrain(:,j) - mean(ftrain(:,j));

            R   = R - ftrain(:,j);
        end
        
        epsilon = max(max(abs(ftrain_old - ftrain)));
        iter    = iter + 1;
        if (verbose)
            fprintf('\t Backfitting: At iteration %d, epsilon is %f\n', iter, epsilon);
        end
    end
    
    % finish fitting on the training data and now perform prediction
    for j=1:p
        ftest(:,j)  = Stest{j}*(R + ftrain(:,j));
    end
    
end
