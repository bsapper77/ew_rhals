function [W,H,step,violations]=randomized_nnmf(A,k,p,alpha,beta,gamma,delta,tolerance,maxsteps,initialize,SigmaMat,sigma,like_ristretto,wtwtilde,pullW,pullH,a,b)






%This is translated from Python function compute_rnmf from Ristretto Folder
%Folder referred to from Erichson '17

%A(matrix): Data Matrix
%k(int): Target rank
%p(int): oversampling
%gamma(int): l2 regularization for W 
%delta(int): l2 regularization for H 
%alpha(int): l1 regularization for W 
%beta(int): l1 regularization for H
%tolerance(int): For stopping the algorithm: If change of cost function
%   divided by the initial value of the cost function is below tolerance
%   for 3 consecutive steps, the algorithm halts
%maxsteps(int): maximum steps of the algorithm
%initialize(string, 3 choices): Initialization of Factor Matrices:
%   random: absolute value of random normals, with mean as the mean of the
%   data matrix divided by the uncertainties
%   nndsvd: positive version of a randomized SVD, negative elements filled
%   to zeros
%   nndsvdmean: positive version of a randomized SVD, negative elements
%   filled with mean of data matrix divided by the uncertainties
%SigmaMat(matrix): Uncertainties Matrix
%sigma(boolean): If true, Weighted NNMF is ran, if false, unweighted NNMF 
%   is ran
%like_ristretto(boolean): If true, in each update, W is found by using 
%   the matrix product Q*B*H' in the update algorithm, as in the ristretto 
%   code, if false, W_tilde is found by using the matrix product BH', and 
%   then W is found by taking the maximum between 0 and Q*W_tilde, as
%   detailed in the pseudocode in Erichson '17
%wtwtilde(boolean): If true, W_tilde'*W_tilde is used in the update rules,
%   if false, W'*W is used in the update rules to enhance the accuracy, as
%   proposed in Erichson '17
%pullW(boolean): If true, pullH is false, and values of W are pulled up
%(values of H are pulled down)
%pullH(boolean): If true, pullW is false, and values of H are pulled up 
%(values of H are pulled down)
% a: Parameter by which to pull H up or down
% b: Parameter by which to pull W up or down

% Objective: min_{W,Ht} ||(R(j)-W(:,j)*Ht(:,j)')./\SigmaMat||_F^2+
%   ||W(:,j)||_1+||Ht(:,j)||_1+||W(:,j)||_2^2+||Ht(:,j)||_2^2
%   where R(j)=A-\sum_{i ~= j}W(:,i)*Ht(:,i)'

%W,H: Factor Matrices
%Step(int): Steps in iteration
%violations(vector of size maxsteps+1): Violation (Projected Gradient 
%   Squared) at each step, with first element being 0th step


proj_grad=true; %Projected Gradient used as stopping condition
proj_back=true; %External Weighting is Ran



check1=false; %For stopping condition
check2=false; %For stopping condition


%Deal with column heavy matrix
flipped=false;
[m,n]=size(A);
if(m<n)
    flipped=false;
    A=A';
    [m,n]=size(A);
end




if(proj_back==true && sigma==true) %If Weighted NNMF is ran
    A1=A./SigmaMat; %Predivide data matrix by uncertainties; ./ refers to elementwise division
else
    A1=A;
end




% Initialization

[W,H]=initializefactors(A1,k,initialize); %Initialize W,H
Ht=H'; %H transpose




%Randomization Step

ell=k + p; % Number of samples to take
Omega=randn(n,ell); % Random vectors
Y=A1*Omega; % Random Vectors moved into the column space of data matrix
[Q,~,~]=qr(Y,0); % Orthonormalization of random vectors
% qr(Y,0) is an economy qr decomposition (if rows>cols, only first n cols 
%   of Q are find, and first n rows of R are found). If rows<=cols, qr and 
%   economy qr are same
B=Q'*A1; % The projection of the data into a lower dimension is B
W_tilde=Q'*W; %Project W into a lower dimension






if(proj_grad==true)
    % Initial Values for Indicators
    %tic
    adj_violation=zeros(1,maxsteps);
    adj_violation(1)=1;
    violations=zeros(1,maxsteps); %Possible speed boost by tracking the 
%   projected gradient of the factor matrices as a stopping condition - not
%   extensively tested
    %time_projgrad(1)=toc;
end




if(conv_by_cost==true)
    %tic
    if(sigma==false) %Not weighted NNMF
        error_converge(1)=norm(A-W*Ht','fro')^2; %Unweighted Error
    else %Weighted NNMF
        error_converge(1)=norm((A-W*Ht')./SigmaMat,'fro')^2; %Weighted Error; ./ refers to elementwise division
    end
    W1cost=sum(sum(W)); %L1 norm of columns of W
    H1cost=sum(sum(Ht)); %L1 norm of rows of H
    W2cost=norm(W,'fro')^2; %L2 norm of columns of W
    H2cost=norm(Ht,'fro')^2; %L2 norm of columns of H
    l1cost(1)=alpha*W1cost+beta*H1cost; %Total L1 cost
    l2cost(1)=gamma*W2cost+delta*H2cost; %Total L2 cost
    cost_converge(1)=error_converge(1)+l1cost(1)+l2cost(1); %Total Cost
    adj_cost(1)=1;
    %time_costcalc(1)=time_costcalc(1)+toc;
end






%Main Algorithm
for step=1:maxsteps
    [W,Ht,W_tilde,violation,tpg]=updateWHrandom(B,W,Ht,Q,alpha,beta,gamma,delta,W_tilde,like_ristretto,wtwtilde,pullW,pullH,a,b,proj_grad); %Run update algorithm
    %time_projgrad(step+1)=tpg;






    
    if(proj_grad==true)
        %tic
        violations(step)=violation;
        adj_violation(step)=violations(step)/violations(1);
        if(step>=2)
            stop_cond=abs(adj_violation(step)-adj_violation(step-1));
        end
        %time_projgrad(step+1)=time_projgrad(step+1)+toc;
    end

    if(step>=2)
        if(stop_cond<=tolerance)
            if(check1==true) %If criteria was met one step before, this is true
                if(check2==true) %If criteria was met two steps before, this is true
                    break %End for loop
                else
                    check2=true; %Condition was true for two previous steps
                end
            else
                check1=true; %Condition was true for the previous step
            end
        else
            check1=false; %Reset checks
            check2=false; %Reset checks
        end
    end
end
disp(step) %Print step number
H=Ht'; %H is transpose of Ht

% Find actual W and H using pseudoinverses

% NOTE: This code is not adapted from any research papers or pre existing
% code. The only motivation is that this code seems to produce feasible
% results and we haven't thought of any better alternatives. More testing
% needs to be done on the convergence of this method to feasible solutions

% The method attempts to solve W*H=(W_alg*H_alg).*SigmaMat, where W_alg and
% H_alg are the factor matrices from the algorithm in which A is predivided
% by SigmaMat (the uncertainties). The term ".*" refers to elementwise 
% multiplication. Thus, this method finds W and H iteratively through 
% W=((W_alg*H_alg).*SigmaMat)*pseudoinverse(H), and H=pseudoinverse(W)*
% ((W_alg*H_alg).*SigmaMat). This method is equivalent to Ordinary Least 
% Squares: H=inv(W'W)W'((W_alg*H_alg).*SigmaMat), with W'
% being transpose of W, and W=((W_alg*H_alg).*SigmaMat)H'inv(HH').




if(proj_back==true && sigma==true) %Weighted NNMF is ran


    iter=20;
    tol_pb=10^(-4);
    change_each_step=zeros(1,iter); % Change in factor matrices throughout the
    %    iterations
    %e_each_step=zeros(1,iter);
    wh=(W*H).*SigmaMat; %Actual factors W*H is related to calculated factor matrices by (W_est*H_est).*SigmaMat
    H_final=sqrt(mean(A,'all')/k)*(H/mean(H,'all')); % Initial H in iteration is 
%    estimated as the values of H scaled to multiply by W to equal the mean of A
    for i=1:iter %Iteration step
        if(i>=2) %If not first step
            W_prev=W_final; % previous W
            H_prev=H_final; % previous H
        else
            W_prev=W; %First W
            H_prev=H_final; %First H
        end
        W_final=wh*pinv(H_final); % Pseudoinverse calculation of W
        W_final(W_final<0)=0; % Non negativity
        H_final=pinv(W_final)*wh; % Pseudoinverse calculation of W
        H_final(H_final<0)=0; % Non negativity
        change_each_step(i)=norm(W_final-W_prev,'fro')/norm(W_prev,'fro')+norm(H_final-H_prev,'fro')/norm(H_prev,'fro'); %Frobenius norm of change in factor matrices divided by frobenius norm of previous factor matrix
        if(change_each_step(i)<tol_pb)
            disp(i) %Print step if convergence is reached
            break %Break out of for loop
        end
        %e_each_step(i)=norm((A-W_final*H_final)./SigmaMat,'fro');
    end
    W=W_final; %New W is W final
    H=H_final;
end

if(flipped==true)
    W1=W; %Store values of W as W1
    W=Ht; %Set W equal to transpose of H
    H=(W1)'; %Set H equal to transpose of W
end

end











