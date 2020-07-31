function [X_norm] = featureNormalize(X)
% Olceklendirme yapilan fonksiyon

X_norm = X;
m=size(X,2);
X_mean= mean(X);


for i = 1:m
    sDerivation=std(X(:,i));
    X_norm(:,i)=(X_norm(:,i) - X_mean(:,i))/sDerivation;
end



end
