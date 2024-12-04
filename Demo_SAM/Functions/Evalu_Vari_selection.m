function [TP,FP,C,U,O]=Evalu_Vari_selection(feature,True_feature,False_feature)
%evaluate the performance of variable selection
%feature: selected features or variables
%True_feature: ground-truth feature
%TP: True positive--number of truly informative variables selected
%FP: False positive--number of truly non-informative variables selected
%C: 1 if correct-fitting
%U: 1 if under-fitting
%O: 1 if over-fitting
S=feature;T=True_feature;F=False_feature;
C=0;U=0;O=0;
TP=ismember(S,T);TP=sum(TP);
FP=ismember(S,F);FP=sum(FP);
if TP==length(T)&&FP==0
    C=1;
elseif TP==length(T)&&FP>0
    O=1;
else
    U=1;
end