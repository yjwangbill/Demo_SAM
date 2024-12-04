function X = scaleData(X)
    % scale the data so that each feature ranges from 0 to 1
    [n, p]  = size(X);
    for j = 1:p
        MIN = min(X(:,j));
        MAX = max(X(:,j));
        X(:,j) = (X(:,j) - MIN)/(MAX - MIN);
    end
end