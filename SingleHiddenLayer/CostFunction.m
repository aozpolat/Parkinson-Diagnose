function [J grad] = CostFunction(nn_params, input_layer_size, hidden_layer_size, X, y, lambda)


% unrolled halde gelen parametrelerin normal haline getirildigi kisim
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), 1, (hidden_layer_size + 1));


m = size(X, 1); %ornek sayisi
         

J = 0;
%forward propagation
firstLayer=[ones(m,1) X];
secondLayer=sigmoid(firstLayer*transpose(Theta1));
secondLayer=[ones(size(secondLayer,1),1) secondLayer];
finalLayer=sigmoid(secondLayer*transpose(Theta2));

Theta1r=Theta1(:,2:end);
Theta2r=Theta2(:,2:end);

J= J + (((1/m )* (transpose(-y)*log(finalLayer)  - transpose(1 - y)*log(1-finalLayer)))) ;        
J= J + (lambda/(2*m))* ( sumsqr(Theta1r)+ sumsqr(Theta2r));


%back propagation
 total_error_1=zeros(size(Theta1));
 total_error_2=zeros(size(Theta2));
 
 
 for i=1:m
     a_1=X(i,:);
     a_1=[1 a_1];
     z_2=a_1*transpose(Theta1);
     a_2=sigmoid(z_2);
     a_2=[1 a_2];
     z_3=a_2*transpose(Theta2);
     a_3=sigmoid(z_3);    
     S_3= a_3-y(i);
     S_2=transpose(Theta2)*S_3 .* transpose([1 sigmoidGradient(z_2)]);
     S_2=S_2(2:end);
     
     total_error_1=total_error_1+ S_2*a_1;
     total_error_2=total_error_2+ S_3*a_2;
 end    

 Theta1_grad=total_error_1 / m;
 Theta1_grad(:,2:end)= Theta1_grad(:,2:end) + (lambda/m).* Theta1(:,2:end);
 Theta2_grad=total_error_2 / m;
 Theta2_grad(:,2:end)= Theta2_grad(:,2:end) + (lambda/m).* Theta2(:,2:end);
  

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
