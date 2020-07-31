function p = predict(Theta1, Theta2, X)

m = size(X, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
hypo = sigmoid([ones(m, 1) h1] * Theta2'); %hipotez
p=hypo>=0.5;

end
