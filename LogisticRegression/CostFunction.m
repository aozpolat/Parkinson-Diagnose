function [J, grad] = CostFunction(theta, X, y, lambda)

m = length(y); 
grad = zeros(size(theta));

hypo= sigmoid(X*theta);  %hipotez
theta2=theta(2:end,:);   %ilk row bias unit o nedenle çıkarılır

J=((1/m )* (transpose(-y)*log(hypo)  - transpose(1 - y)*log(1-hypo))) ...
    +(lambda/(2*m))* sum(theta2 .^2);

grad= ((transpose(X)* (hypo-y)) ./ m);
grad(2:end)= grad(2:end) + (lambda/m).* theta2;



end
