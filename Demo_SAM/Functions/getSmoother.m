%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author  :   Junming Yin
% Contact :   junmingy@email.arizona.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function S = getSmoother(Xtrain, X, option)
    % pre-compute the smoother matrix
    % Xtrain:   training data matrix, size n*p
    % X:        a data matrix, size m*p
    % S:        each cell is a smoother matrix between X and Xtrain (size m*n) for a particular coordinate
    
    if ~exist('option','var') || ~isfield(option, 'ktype')
        ktype = 'gauss';      % default kernel: Gaussian
    else
        ktype = option.ktype;
    end
    
    [n, p]  = size(Xtrain);  
    m       = size(X, 1);

    S       = cell(p,1);  % each cell of S is an m by n matrix

    for j = 1:p
        % plug-in bandwidth
        bandwidth   = option.scaler * std(Xtrain(:,j)) * n^(-1/5);
        % each S{j}: (m by n) matrix
% %         S{j}    = abs(repmat(X(:,j), 1, n) - repmat(Xtrain(:,j)', m, 1))./bandwidth;
        S{j}    = abs(bsxfun(@minus, X(:,j), Xtrain(:,j)'))./bandwidth;

        switch ktype
            case 'gauss'
                % Gaussian kernel
                S{j}    = gaussianKernel(S{j});
            case 'tgauss'
                % truncated Gaussian kernel
                S{j}    = truncated_Gaussian(S{j});
            case 'epan'
                % Epanechnikov quadratic kernel
                S{j}    = epanechnikov(S{j});
        end
        
        % normalize the row of smoother matrices
        S{j}    = normalizeRows(S{j});
        
        if strcmp(ktype, 'gauss') ~= 1
            % sparsify the smoother matrix
            S{j}    = sparse(S{j});
        end
    end    
end


function A = normalizeRows(A)
    % normalize each row of A so that the row sum is 1
% %     nCol    = size(A, 2);
% %     A       = A .* repmat(1./sum(A, 2), 1, nCol);
    A       = bsxfun(@rdivide, A, sum(A,2));
end

function K = gaussianKernel(x)
    % Gaussian kernel
    K       = normpdf(x, 0, 1);
end

function K = truncated_Gaussian(x)
    % truncated Gaussian kernel
    K       = exp(-(x.^2)./2);
    K       = (K >= exp(-0.5)).*K;
end

function K = epanechnikov(x)
    % Epanechnikov quadratic kernel
    K       = max(0, 0.75.*(1 - x.^2));
end
