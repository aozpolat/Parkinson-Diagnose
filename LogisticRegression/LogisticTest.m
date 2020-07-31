function meanResults = LogisticTest(X,y)
result=zeros(500,1);
meanResults=zeros(1,1);
    
   for i=1:500
        rand=randperm(195,39)'; %random 39 eleman secilir
        X_tmp=X;
        y_tmp=y;
        X_test=X_tmp(rand,:);
        X_tmp(rand,:)=[];
        X_train=X_tmp;
        y_test=y_tmp(rand,:);
        y_tmp(rand,:)=[];
        y_train=y_tmp;
        
        lambda = 0.25;
        [all_theta] = idealThetas(X_train, y_train, lambda);    
        pred = predict(all_theta, X_test);
        
       % result(i,1)=t;
        result(i,1)= mean(double(pred == y_test)) * 100;     
  end
  %meanResults(index,1)=result(1,1);
   meanResults=mean(result);
      
    
    
    
end

