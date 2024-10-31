function [fTest]=CSPfeature_test(xTest,W)

fTest=zeros(size(xTest,3),size(W,2));
for i=1:size(xTest,3)
    X=W'*xTest(:,:,i);
    fTest(i,:)=log10(diag(X*X')/trace(X*X'));
end