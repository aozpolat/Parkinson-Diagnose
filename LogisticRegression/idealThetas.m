function [ideal_theta] = idealThetas(X, y,lambda)

[m,n] = size(X); 

X = [ones(m, 1) X]; %bias unit ekleme

initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50); 
theta =fmincg (@(t)(CostFunction(t, X, y, lambda)), initial_theta, options);   
ideal_theta=transpose(theta);
   
end
