%% Adaptive Linear Discriminant Analysis : run functions

[error, IDX] = AdaLDA(train_LW,train_HW,test,label_test,lambda0)

[accuracy] = AdaLDA_accuracy(IDX,label_test)
    

   