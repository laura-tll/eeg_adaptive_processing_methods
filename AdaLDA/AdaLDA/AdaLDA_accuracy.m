%% Adaptive Linear Discriminant Analysis : accuracy 

function accuracy = accuracy(IDX,label_test);
c=0;
[m,n]=size(label_test);
for i=1:n;
    if (IDX(i)==label_test(i))
        c=c+1;
    end
end
accuracy = c/n;