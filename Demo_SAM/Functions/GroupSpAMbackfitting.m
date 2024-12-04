%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author  :   Junming Yin
% Contact :   junmingy@email.arizona.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f0 fnorm feature] = GroupSpAMbackfitting(Y, S, Group, lambda, option)

    maxiter     = option.maxiter;
    tol         = option.tol;
    verbose     = option.verbose; 
    criterion   = option.criterion;
    
    n           = length(Y);
    p           = length(S);
    nG          = length(Group);        % number of groups
    f0          = mean(Y);              % intercept
    fhat        = zeros(n, p);          % fitted values at training data points
    fnorm       = sqrt(mean(fhat.^2));  % estimation of L2 norm of function f_j; l2 norm(fhat)/sqrt(n)
    R           = Y - f0;               % residuals
    
    epsilon     = 1;
    iter        = 0;
    while (epsilon > tol && iter < maxiter)
        fhat_old= fhat;
        ind     = randperm(nG);
        for jj  = 1 : nG
            currentG            = Group{ind(jj)};    % a set (group)
            R                   = R + sum(fhat(:, currentG), 2);
            
            fhat(:,currentG)    = GroupSpAMthresholding(currentG, R, S(currentG), lambda, option);
            
            R                   = R - sum(fhat(:, currentG), 2);
        end
     
        fnorm   = sqrt(mean(fhat.^2));
        switch criterion
            case 1  % maximum difference
                epsilon = max(max(abs(fhat_old - fhat)));
            case 2  % mean difference
                epsilon = mean(mean(abs(fhat_old - fhat)));
            case 3  % relative difference
                epsilon = norm(fhat_old - fhat, 'fro')/norm(fhat_old, 'fro');
        end
        iter    = iter + 1;
        if (verbose)
            fprintf('\t GroupSpAMbackfitting: At iteration %d, epsilon is %f\n', iter, epsilon);
        end
    end
    
    feature     = find(fnorm > eps);
end
