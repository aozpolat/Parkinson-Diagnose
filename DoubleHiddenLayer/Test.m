function meanResults = Test(X,y,first_hidden_layer_size,second_hidden_layer_size)
%random sekilde ilk agirliklar belirlenir
initial_Theta1=randInitializeWeights(22, first_hidden_layer_size);
initial_Theta2 = randInitializeWeights(first_hidden_layer_size, second_hidden_layer_size);
initial_Theta3 = randInitializeWeights(second_hidden_layer_size, 1);

initial_unrolled_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

result=zeros(500,1);
%meanResults=zeros(1,2);
for k=39:39      
    for i=1:500
        rand=randperm(195,k)'; %random 39 row belirlenir.
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
        costFunction = @(p) CostFunction(p, 22, first_hidden_layer_size,second_hidden_layer_size, X_train, y_train, lambda);
        [unrolled_params, ~] = fmincg(costFunction, initial_unrolled_params, options);
        
        Theta1 = reshape(unrolled_params(1:first_hidden_layer_size * (22 + 1)),first_hidden_layer_size, (22 + 1));
        Theta2 = reshape(unrolled_params((1 + (first_hidden_layer_size * (22 + 1))):end-second_hidden_layer_size-1), ...
                second_hidden_layer_size, (first_hidden_layer_size + 1));     
        Theta3 = reshape(unrolled_params(end-second_hidden_layer_size:end), 1,(second_hidden_layer_size +1)); 
        pred = predict(Theta1, Theta2,Theta3, X_test);
        
        %result(i,1)=k;
        result(i,1)= mean(double(pred == y_test)) * 100;      
    end
    %meanResults(index,1)=result(1,1);
    %meanResults(index,2)=mean(result(:,2));
    meanResults=mean(result); 
      
  
    
end
end
