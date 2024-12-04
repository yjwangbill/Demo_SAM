%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author  :   Junming Yin
% Contact :   junmingy@email.arizona.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [score, componentNorms, bestLambda, bestFeature] = GroupSpAMselection(Ytrain, Strain, Yvalid, Svalid, Group, option)
% score:            l2 loss function
% componentNorms:   empirical estimation of L2 norm for each component function
% bestLambda:       chosen lambda 
% bestFeature:      chosen features

%------------------------------------------------------------------- 

lambda          = option.lambda;
n               = length(Ytrain);
p               = length(Strain);
nLambda         = length(lambda);
score           = inf(nLambda, 1);
features        = cell(nLambda, 1);
componentNorms  = zeros(nLambda, p);

%------------------------------------------------------------------- 

for i = 1:nLambda
    fprintf('\t Current lambda: %f\n', lambda(i));
    [f0 fnorm features{i}]          = GroupSpAMbackfitting(Ytrain, Strain, Group, lambda(i), option); 
    componentNorms(i, :)            = fnorm;
        
    if length(features{i}) > 0
        % backfitting on the selected features
        [f0, ftrain, fvalid, R]     = backfitting(Ytrain, Strain(features{i}), Svalid(features{i}), option);
        % MSE on the validation set
        score(i)                    = mean( (sum(fvalid, 2) + f0 - Yvalid).^2 );
    else
        % MSE on the validation set
        score(i)                    = mean( (f0 - Yvalid).^2 );
    end
end

[dumm I]    = min(score);
bestLambda  = lambda(I);
bestFeature = features{I};

% fprintf('Best lambda for GroupSpAM: %f\n', bestLambda);
% fprintf('Best Feature of GroupSpAM: %s\n', int2str(bestFeature));
