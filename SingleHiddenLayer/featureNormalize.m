function X_norm = featureNormalize(X)

%verinin normalize edildigi fonksiyon

X_norm = X;
n=size(X,2); %sutun sayisi
X_mean= mean(X);


for i = 1:n
    sDeviation=std(X(:,i));  
    X_norm(:,i)=(X_norm(:,i) - X_mean(:,i))/sDeviation;
end

end
