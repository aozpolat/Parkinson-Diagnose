function meanResults = Test(X,y,hidden_layer_size)


initial_Theta1 = randInitializeWeights(22, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, 1);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

result=zeros(500,1); %ortalamasi alinacak degerlerin tutuldugu matris.
%meanResults=zeros(50,2);
for k=39:39
    
    for i=1:500
        rand=randperm(195,k)';
        X_tmp=X;
        y_tmp=y;
        X_test=X_tmp(rand,:);
        X_tmp(rand,:)=[];
        X_train=X_tmp;
        y_test=y_tmp(rand,:);
        y_tmp(rand,:)=[];
        y_train=y_tmp;
        
        options = optimset('MaxIter', 50);
        lambda = 0.25;
        costFunction = @(p) CostFunction(p, 22, hidden_layer_size, X_train, y_train, lambda);
        [nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

        Theta1 = reshape(nn_params(1:hidden_layer_size * (22 + 1)), hidden_layer_size, (22 + 1));
        Theta2 = reshape(nn_params((1 + (hidden_layer_size * (22 + 1))):end), 1, (hidden_layer_size + 1));
 
        pred = predict(Theta1, Theta2, X_test);
        %result(i,1)=t;       
        result(i,1)= mean(double(pred == y_test)) * 100;      
        
    end
    %meanResults(index,1)=result(1,1);
    %meanResults(index,2)=mean(result(:,2));
    meanResults=mean(result); %bulunan butun degerlerin ortalamasi alinir
     
end
end

