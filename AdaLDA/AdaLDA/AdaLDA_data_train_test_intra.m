%% Adaptive Linear Discriminant Analysis : seperating training and test data (intra-subject)

% final_data : matrice of all the data of all participants 



LW = []; % train data for LW 
train_LW_f = [];

HW = []; % train data for HW 
train_LW_f = [];

test = [];  % test data

acc_all = []
for k=(1:13);
    k
    acc = 0;
    % training data  
    LW = cat(1,reshape(final_data(k,1:158,:),[158,60]));
    HW = cat(1,reshape(final_data(k,159:316,:),[158,60]));
    
    train_LW = cell(1,3);
    train_LW{1} = LW(1:52,:);
    train_LW{2} = LW(27:79,:);
    train_LW{3} = cat(1,LW(1:26,:),LW(53:79,:));
        
    train_HW = cell(1,3);
    train_HW{1} = HW(1:52,:);
    train_HW{2} = HW(27:79,:);
    train_HW{3} = cat(1,HW(1:26,:),HW(53:79,:));
       
    
    % testing data 
    test_LW = cell(1,3);
    test_LW{1} = LW(79:130,:);
    test_LW{2} = LW(105:157,:);
    test_LW{3} = cat(1,LW(79:104,:),LW(131:157,:));
    
    test_HW = cell(1,3);
    test_HW{1} = HW(79:130,:);
    test_HW{2} = HW(105:157,:);
    test_HW{3} = cat(1,HW(79:104,:),HW(131:157,:));

        
    train_LW_f = [];
    train_HW_f = [];
    
    for ltrain=(1:3);
        for htrain=(1:3);
            train_LW_f = train_LW{ltrain};
            train_HW_f = train_HW{htrain};
            
           
            for ltest=(1:3);
                for htest=(1:3);
                    test_f = cat(1,test_LW{ltest},test_HW{htest});
                    [m,n]=size(test_LW{ltest});
                    x = zeros(1,m);
                    [m,n]=size(test_HW{htest});
                    y = ones(1,m);
                    label_test = cat(2, x, y); % label for test data 
                    label_test = int64(label_test);
                    
                    [error, IDX] = AdaLDA(train_LW_f,train_HW_f,test_f,label_test,lambda0);
                    acc1 = AdaLDA_accuracy(IDX,label_test);
                    acc = acc + acc1;
                    
                end
            end
        end
    end
    
accuracy = acc;
acc_all = [acc_all,accuracy]
    
end 

                
                
                
                
            