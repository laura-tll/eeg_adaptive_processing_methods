%% Adaptive Linear Discriminant Analysis : seperating training and test data (inter-subject)

% final_data : matrice of all the data of all participants 

acc = 0;
lambda0=1;
x = zeros(1,158);
y = ones(1,158);
label_test = cat(2, x, y, x, y); % label for test data 
label_test = int64(label_test);



for i=(1:13);
    for j=(i+1:13);
       
        train_LW = []; % train data for LW 
        train_HW = []; % train data for HW 
        test = [] ; % test data

        for k=(1:13);

            if k==i || k==j; % test data : all participants combinations (of 2) for testing 
                test = cat(1,test,reshape(final_data(k,:,:),[316,60]));
                
            else % train data : other participants than the first two chosen for the test 
                 train_LW = cat(1,train_LW,reshape(final_data(k,1:158,:),[158,60]));
                 train_HW = cat(1,train_HW,reshape(final_data(k,159:316,:),[158, 60]));
            end
        end
        
        [error, IDX] = AdaLDA(train_LW,train_HW,test,label_test,lambda0);
        
        [accuracy] = AdaLDA_accuracy(IDX,label_test)
        acc = acc + [accuracy];
        
    end
end

acc_final = acc/78
                
                
                
                
            