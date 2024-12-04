%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author  :   Junming Yin
% Contact :   junmingy@email.arizona.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  fhat = GroupSpAMthresholding(currentG, R, S, lambda, option)

    maxiter     = option.maxiter;
    tol         = option.tol;
    criterion   = option.criterion;
    
    n           = size(S{1}, 1);
    dg          = length(currentG);  	% size of group
    QR          = zeros(n*dg, 1);
    J           = zeros(n*dg, n*dg);
    curnorm     = 0;

    % compute the norm and form the matrices for equations
    for j = 1:dg
        sRow = (j - 1)*n + 1;
        eRow = j*n;
        P = S{j}*R;
        QR(sRow:eRow) = P;
        for i = 1:dg
            sCol = (i - 1)*n + 1;
            eCol = i*n;
            if (i == j)
                J(sRow:eRow, sCol:eCol) = eye(n);
            else
                J(sRow:eRow, sCol:eCol) = S{j};
            end
        end
        curnorm = curnorm + (norm(P)^2)/n;
    end

    curnorm     = sqrt(curnorm);

    if curnorm <= lambda*sqrt(dg)
        fhat  = zeros(n*dg, 1);
    else
        % solve the equation by fixed point iteration
        iter    = 0;
        epsilon = 1;
        fhat    = rand(n*dg, 1);
        A       = eps*eye(n*dg);
        B       = lambda*sqrt(n*dg)*eye(n*dg);
        while (epsilon > tol && iter < maxiter)
            fhat_old    = fhat;
            fhat        = (J + A + B/norm(fhat_old)) \ QR;
            switch criterion
                case 1      % maximum difference
                    epsilon     = max(abs(fhat_old - fhat));
                case 2      % mean difference
                    epsilon     = mean(abs(fhat_old - fhat));
                case 3      % relative difference
                    epsilon     = norm(fhat_old - fhat)/norm(fhat_old);
            end
            iter        = iter + 1;
        end
    end

    fhat    = reshape(fhat, n, dg);
    fhat    = fhat - repmat(mean(fhat), n, 1);
end
