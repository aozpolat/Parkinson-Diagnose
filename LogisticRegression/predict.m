function p = predict(all_theta, X)

m = size(X, 1);

X = [ones(m, 1) X];      
z=sigmoid(X*transpose(all_theta));

p=z>=0.5;


end
