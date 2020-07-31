function [J grad] = CostFunction(unrolled_params, input_layer_size,first_hidden_layer_size,second_hidden_layer_size, ...
                                   X, y, lambda)
                               
%unrolled gelen parametrelerin acildigi kisim                              
Theta1 = reshape(unrolled_params(1:first_hidden_layer_size * (input_layer_size + 1)), ...
                 first_hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(unrolled_params((1 + (first_hidden_layer_size * (input_layer_size + 1))):end-second_hidden_layer_size-1), ...
                second_hidden_layer_size, (first_hidden_layer_size + 1));
             
Theta3 = reshape(unrolled_params(end-second_hidden_layer_size:end), 1,(second_hidden_layer_size +1));            


m = size(X, 1); %örnek sayısı       
J = 0;
Theta1_grad = zeros(size(Theta1));  %girdi katmanı ve 1. gizli katman arasındaki ağırlık matrisi
Theta2_grad = zeros(size(Theta2));  %1. ve 2. gizli katman arasındaki ağırlık matrisi
Theta3_grad = zeros(size(Theta3));  %2. gizli katman ve çıktı katmanı arasındaki ağırlık matrisi

%Kademeli şekilde çıktı katmanına gidilen kısım
firstLayer=[ones(m,1) X];
secondLayer=sigmoid(firstLayer*transpose(Theta1));
secondLayer=[ones(size(secondLayer,1),1) secondLayer];
thirdLayer=sigmoid(secondLayer*transpose(Theta2));
thirdLayer=[ones(size(thirdLayer,1),1) thirdLayer];
finalLayer=sigmoid(thirdLayer*transpose(Theta3));

%bias term çıkarılıyor
Theta1r=Theta1(:,2:end);
Theta2r=Theta2(:,2:end);
Theta3r=Theta3(:,2:end);

 J= J + (((1/m )* (transpose(-y)*log(finalLayer)  - transpose(1 - y)*log(1-finalLayer)))) ;
      
  %regülarizasyon 
 J= J + (lambda/(2*m))* ( sumsqr(Theta1r)+ sumsqr(Theta2r) + sumsqr(Theta3r));
 

 totalError_1=zeros(size(Theta1));
 totalError_2=zeros(size(Theta2));
 totalError_3=zeros(size(Theta3));
 for i=1:m
     %her bir örnek için ayrı ayrı hesaplama yapılır
     a_1=X(i,:);
     a_1=[1 a_1];
     z_2=a_1*transpose(Theta1);
     a_2=sigmoid(z_2);
     a_2=[1 a_2];
     z_3=a_2*transpose(Theta2);
     a_3=sigmoid(z_3);
     a_3=[1 a_3];
     z_4=a_3*transpose(Theta3);
     a_4=sigmoid(z_4);
     
     %hata hesaplamaları
     S_4= a_4-y(i);
     S_3=transpose(Theta3)*S_4.* transpose([1 sigmoidGradient(z_3)]);
     S_3=S_3(2:end);
     S_2=transpose(Theta2)*S_3.* transpose([1 sigmoidGradient(z_2)]);
     S_2=S_2(2:end);
     
     %toplam hataların bulunması
     totalError_1=totalError_1+ S_2*a_1;
     totalError_2=totalError_2+ S_3*a_2;
     totalError_3=totalError_3+ S_4*a_3;
 end    
 %grad değerlerinin belirlenmesi
 Theta1_grad=totalError_1 / m;
 Theta1_grad(:,2:end)= Theta1_grad(:,2:end) + (lambda/m).* Theta1(:,2:end);
 Theta2_grad=totalError_2 / m;
 Theta2_grad(:,2:end)= Theta2_grad(:,2:end) + (lambda/m).* Theta2(:,2:end);
 Theta3_grad=totalError_3/m;
 Theta3_grad(:,2:end)=Theta3_grad(:,2:end) + (lambda/m) .* Theta3(:,2:end);
 
 %tek bir vektöre eklenir.
 grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
