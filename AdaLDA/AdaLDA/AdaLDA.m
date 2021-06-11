%% Adaptive Linear Discriminant Analysis
% It is based on the constrained l_1 minimization to classify a
% p-dimensional vecotr to one of two multivariate Gaussian distributions
% with the same covariance matrix.

%% Outputs:
% error: estimated parameters for the Gaussian mixtures
% IDX: vector of class membership
% Inputs: 
% xt: n by p training data from class 1
% yt: n by p training data from class 2
% ztest: n_z by p testing data drawn from one of two classes
% label_z: true labels of the test data, used for evaluating the
%   classificatoin performance.
% lambda0: a scalar, the tuning parameter for estimating sparse beta. Default
%    is 1. Different choice may lead to better finite sample peformance.
%




function [error, IDX] = AdaLDA(train_LW,train_HW,test,label_test,lambda0);

lambda0=1;
[n,p]=size(train_LW);
hatmu_LW=mean(train_LW);  %estimation of mean
hatmu_HW=mean(train_HW);

hatmu_LW_f = permute(hatmu_LW,[2 1 3]);
hatmu_HW_f = permute (hatmu_HW, [2 1 3]);

mu = [hatmu_LW_f, hatmu_HW_f];
hatdelta=hatmu_LW_f-hatmu_HW_f;
hatSigma=(cov(train_LW)+cov(train_HW))/2; %estimation of covariance matrix
%% Step 1 of the AdaLDA procedure
d=diag(hatSigma);
a=1*sqrt(log(p)/n)*(sqrt(d));
B=lambda0*sqrt(log(p)/n)*sqrt(d)*hatdelta';
f=ones(2*p,1);
CoeffM=[hatSigma-B -(hatSigma-B);-(hatSigma+B) hatSigma+B];
Coeffb=[a+hatdelta; a-hatdelta];
options=optimoptions('linprog','Algorithm','interior-point','Display','none');
uv=linprog(f,CoeffM,Coeffb,[],[],zeros(2*p,1),[],[],options);
beta0=uv(1:p)-uv((p+1):(2*p));
%% Step 2 of the AdaLDA procedure
lambda=[];
for k=1:p
    lambda=[lambda, 1*sqrt(log(p-1)/n)*sqrt(lambda0*hatSigma(k,k)*(abs(beta0'*hatdelta)+1))];
end
CoeffMnew=[hatSigma -(hatSigma);-(hatSigma) hatSigma];
Coeffbnew=[lambda'+hatdelta;lambda'-hatdelta];
uv_new=linprog(f,CoeffMnew,Coeffbnew,[],[],zeros(2*p,1),[],[],options);
beta=uv_new(1:p)-uv_new((p+1):(2*p));
%% Evaluation on the testing data
IDX = ( (test - ones(size(test,1),1)*mean(mu, 2)')*beta <=1e-06 ); %classification

IDX=int64(IDX)';
error=sum(abs(IDX-label_test))/size(test,1);
end