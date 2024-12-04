function value=calcRSSE(y,pre)
%y--true label vector
%pre--prediction
y=y(:);pre=pre(:);
my=mean(y);My=repmat(my,size(y,1),1);
value=1-sum((y-pre).^2)/sum((y-My).^2);