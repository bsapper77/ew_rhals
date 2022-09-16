function [W,H,step,violations,change_each_step,e_each_step,error_converge,cost_converge,adj_cost,l1cost,l2cost]=randomized_nnmf(A,k,p,alpha,beta,gamma,delta,tolerance,maxsteps,initialize,SigmaMat,sigma,like_ristretto,wtwtilde,pullW,pullH,a,b,range)
rng(range)

conv_by_cost=false;
proj_grad=true;
proj_back=true;
rank_one_scaling=false;


%,time_projgrad,time_costcalc
%time_projgrad=zeros(1,maxsteps+1);
%time_costcalc=zeros(1,maxsteps+1);





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

% Objective: min_{W,Ht} ||(R(j)-W(:,j)*Ht(:,j)')./\SigmaMat||_F^2+
%   ||W(:,j)||_1+||Ht(:,j)||_1+||W(:,j)||_2^2+||Ht(:,j)||_2^2
%   where R(j)=A-\sum_{i ~= j}W(:,i)*Ht(:,i)'





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


if(rank_one_scaling==true && sigma==true)
    Weights=ones(size(A))./SigmaMat;
%     [U,S,V] = LOCAL_rsvd(Weights,1,20);
    vec_B=ones(1,m);
    C=zeros(n,1);
    for its=1:2
        for j=1:n
            C(j)=vec_B*Weights(:,j)/(vec_B*vec_B');
        end
        for i=1:m
            vec_B(i)=Weights(i,:)*C/(C'*C);
        end
    end
%     vec_B=(sqrt(S)*abs(U))';
%     C=sqrt(S)*abs(V);
    A1=A;
    A=repmat(vec_B',1,n).*A.*repmat(C',m,1);
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






if(conv_by_cost==true)
    %Initialize Indicators
    %tic
    cost_converge=zeros(1,maxsteps+1); % Value of cost function
    adj_cost=zeros(1,maxsteps+1); % Adjusted Cost: Cost function divided by the initial value
    error_converge=zeros(1,maxsteps+1); % (Weighted) Squared Residual of
    %   Algorithm
    % violations=zeros(1,maxsteps+1); %Possible speed boost by tracking the
    %   projected gradient of the factor matrices as a stopping condition - not
    %   extensively tested
    l1cost=zeros(1,maxsteps+1); %L1 cost
    l2cost=zeros(1,maxsteps+1); %L2 cost
    %time_costcalc(1)=toc;
end





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
init=false; %Not initialization
for step=1:maxsteps
    [W,Ht,W_tilde,violation,tpg]=updateWHrandom(B,W,Ht,Q,init,alpha,beta,gamma,delta,W_tilde,like_ristretto,wtwtilde,pullW,pullH,a,b,proj_grad); %Run update algorithm
    %time_projgrad(step+1)=tpg;






    if(conv_by_cost==true)
        %tic
        %Calculate Diagnostics
        if(sigma==false) %Unweighted NNMF
            error_converge(step+1)=norm(A1-W*Ht','fro')^2; %Frobenius Norm of Residual
        else %Weighted NNMF
            error_converge(step+1)=norm(A1-(W*Ht')./SigmaMat,'fro')^2; %Frobenius Norm of Weighted Residual
        end
        W1cost=sum(sum(W)); %L1 norm of columns of W
        H1cost=sum(sum(Ht)); %L1 norm of rows of H
        W2cost=norm(W,'fro')^2; %L2 norm of columns of W
        H2cost=norm(Ht,'fro')^2; %L2 norm of rows of H
        l1cost(step+1)=alpha*W1cost+beta*H1cost; %Total L1 cost
        l2cost(step+1)=gamma*W2cost+delta*H2cost; %Total L2 cost
        cost_converge(step+1)=error_converge(step+1)+l1cost(step+1)+l2cost(step+1); %Total cost is sum of error, l1, l2 costs
        adj_cost(step+1)=cost_converge(step+1)/cost_converge(1); %Adjusted cost: Cost over initial cost
        %Stopping Condition
        stop_cond=abs(adj_cost(step+1)-adj_cost(step)); %Change in adjusted cost is the stopping condition
        %time_costcalc(step+1)=toc;
    end





    
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
% ((W_alg*H_alg).*SigmaMat). Why don't we solve this equation using least
% squares? Limited testing has shown using the pseudoinverse to be as 
% accurate and faster than least squares (Least squares would also be an
% iterative approach of H=inv(W'W)W'((W_alg*H_alg).*SigmaMat), with W'
% being transpose of W, and a similar calculation for W). Again, it would
% be interesting to test both in cases with a different number of factors


if(rank_one_scaling==true && sigma==true)
    invB=ones(size(vec_B'))./vec_B';
    W=repmat(invB,1,k).*W;
    invC=ones(size(C'))./C';
    H=H.*repmat(invC,k,1);
end



if(proj_back==true && sigma==true) %Weighted NNMF is ran
    %wh=(W*H).*SigmaMat;
    %[W_final,H_final]=exactnnmf(wh,k,1*10^(-5),1*10^(-5),1*10^(-3),1*10^(-3),tolerance,maxsteps,initialize,SigmaMat,false,pullW,pullH,a,b,range);
    %[W_final,H_final]=randomized_nnmf(wh,k,p,0,0,0,0,tolerance,maxsteps,initialize,SigmaMat,false,like_ristretto,wtwtilde,pullW,pullH,a,b,range);
    %[W_final,H_final]=mult_update(wh,k,0,0,0,0,10^(-5),maxsteps,'random',SigmaMat,false,range);
    %W=W_final;
    %H=H_final;


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

































% function [W,H,step,error_converge,cost_converge,adj_cost,l1cost,l2cost]=randomizednnmf(A,k,p,alpha,beta,gamma,delta,tolerance,maxsteps,initialize,SigmaMat,sigma,like_ristretto,wtwtilde)
% %This is translated from Python function compute_rnmf from Ristretto Folder
% %Folder linked from Erichson '18
% 
% %A: matrix, k: target rank, p: oversampling, gamma: l2 regularization for
% %W, delta: l2 regularization for H, alpha: l1 regularization for W, beta:
% %l1 regularization for H, tolerance: for stopping algorithm, maxsteps:
% %maximum steps of the algorithm
% 
% expectation_maximization=false;
% 
% % If matrix is column heavy
% flipped=false;
% [m,n]=size(A);
% if(m<n)
%     flipped=true;
%     A=A';
%     [m,n]=size(A);
% end
% 
% % Initialization
% [W,H]=initializefactors(A,k,initialize);
% % [W_tilde,~]=initializefactors(B,k,initialize);
% Ht=H';
% if(initialize=="random")
%     mA=mean(mean(A));
%     W=mA*W;
%     Ht=mA*Ht;
% end
% 
% %Randomization Step
% ell       = k + p;
% Omega     = randn(n,ell);
% Y         = A*Omega;
% [Q,~,~]   = qr(Y,0); %qr(Y,0) is an economy qr decomposition (if rows>cols, only first n cols of Q are find, and first n rows of R are found)
% %If rows<=cols, qr and economy qr are same
% if(expectation_maximization==false)
%     B         = Q'*A;
% end
% W_tilde   = Q'*W;
% 
% %Projecting errors into a lower dimension
% % SigmaMat_small=Q'*SigmaMat;
% % SigmaMat_large=SigmaMat;
% 
% %Initialize Indicators
% cost_converge=zeros(1,maxsteps+1);
% adj_cost=zeros(1,maxsteps);
% error_converge=zeros(1,maxsteps+1);
% % violations=zeros(1,maxsteps+1);
% l1cost=zeros(1,maxsteps+1);
% l2cost=zeros(1,maxsteps+1);
% 
% % Initial Values for Indicators
% %  init=true;
% %  [~,~,~,violation_init]=updateWHrandom(B,W,Ht,Q,init,alpha,beta,gamma,delta,SigmaMat_large,SigmaMat_small,W_tilde,like_ristretto,wtwtilde);
% if(sigma==false)
%     error_converge(1)=norm(A-W*Ht','fro')^2;
% else
%     error_converge(1)=norm((A-W*Ht')./SigmaMat,'fro')^2;
% end
% % violation(1)=violation_init;
% W1cost=sum(sum(W));
% H1cost=sum(sum(Ht));
% W2cost=norm(W,'fro')^2;
% H2cost=norm(Ht,'fro')^2;
% l1cost(1)=alpha*W1cost+beta*H1cost;
% l2cost(1)=gamma*W2cost+delta*H2cost;
% cost_converge(1)=error_converge(1)+l1cost(1)+l2cost(1);
% if(alpha==0 && beta==0)
%     l1cost(1)=W1cost+H1cost;
% end
% if(gamma==0 && delta==0)
%     l2cost(1)=W2cost+H2cost;
% end
% 
% %Expectation Maximum Calculation
% if(expectation_maximization==true)
%     weights1=1./SigmaMat;
% end
% 
% %Main Algorithm
% init=false;
% for step=1:maxsteps
%     
%     %Compress A via Expectation Maximization Technique, otherwise run step
%     if(expectation_maximization==true)
%        sigma=false;
%        Acomp=weights1.*A+(ones(size(A))-weights1).*(W*Ht');
%        B=Q'*Acomp;
%        [W,Ht,W_tilde,violation]=updateWHrandom(B,W,Ht,Q,init,alpha,beta,gamma,delta,W_tilde,like_ristretto,wtwtilde);
%     else
%        [W,Ht,W_tilde,violation]=updateWHrandom(B,W,Ht,Q,init,alpha,beta,gamma,delta,W_tilde,like_ristretto,wtwtilde);
%     end
%     
% %     %Violation
% %     if(step==1)
% %         violation_init=violation;
% %         if(violation==0)
% %             print("error")
% %         end
% %     end
% %     
%     %Calculate Indicators
% %     if(sigma==false)
% %         error_converge(step+1)=norm(A-W*Ht','fro')^2;
% %     else
% %         error_converge(step+1)=norm((A-W*Ht')./SigmaMat,'fro')^2;
% %     end
% %     W1cost=sum(sum(W));
% %     H1cost=sum(sum(Ht));
% %     W2cost=norm(W,'fro')^2;
% %     H2cost=norm(Ht,'fro')^2;
% %     l1cost(step+1)=alpha*W1cost+beta*H1cost;
% %     l2cost(step+1)=gamma*W2cost+delta*H2cost;
% %     cost_converge(step+1)=error_converge(step+1)+l1cost(step+1)+l2cost(step+1);
% %     adj_cost(step+1)=cost_converge(step+1)/cost_converge(1);
% %     if(alpha==0 && beta==0)
% %         l1cost(step+1)=W1cost+H1cost;
% %     end
% %     if(gamma==0 && delta==0)
% %         l2cost(step+1)=W2cost+H2cost;
% %     end
% %     violations(step+1)=violation;
%     
% 
%     if(sigma==false)
%         error_converge(step+1)=norm(A-W*Ht','fro')^2;
%     else
%         error_converge(step+1)=norm((A-W*Ht')./SigmaMat,'fro')^2;
%     end
%     W1cost=sum(sum(W));
%     H1cost=sum(sum(Ht));
%     W2cost=norm(W,'fro')^2;
%     H2cost=norm(Ht,'fro')^2;
%     l1cost(step+1)=alpha*W1cost+beta*H1cost;
%     l2cost(step+1)=gamma*W2cost+delta*H2cost;
%     cost_converge(step+1)=error_converge(step+1)+l1cost(step+1)+l2cost(step+1);
%     adj_cost(step+1)=cost_converge(step+1)/cost_converge(1);
%     if(alpha==0 && beta==0)
%         l1cost(step+1)=W1cost+H1cost;
%     end
%     if(gamma==0 && delta==0)
%         l2cost(step+1)=W2cost+H2cost;
%     end
% 
% 
% 
%     %Stopping Condition
%     stop_cond=abs(adj_cost(step+1)-adj_cost(step));
%     if(stop_cond<=tolerance) %If chg(J)/J_0 <= tolerance
%         if(check1==true) %If one time before, this was true
%             if(check2==true) %If two times before, this was true
%                 break
%             else
%                 check2=true;
%             end
%         else
%             check1=true;
%         end
%     else
%         check1=false; %Reset checks
%         check2=false; %Reset checks
%     end
% end
% step
% W1=W;
% H=Ht';
% if(flipped==true)
%     W=Ht;
%     H=(W1)';
% end
% % Find actual W and H
% A1=A.*SigmaMat;
% H_0=sqrt(mean(A1,'all')/k)*(H/mean(H,'all'));
% %H_final=sqrt(mean(A_small,'all')/k)*(0.5*ones(size(H_exact))+rand(size(H_exact))); %H is initiated near the values of the actual H
% change_each_step=zeros(1,20);
% wh=(W*H).*SigmaMat;
% if(sigma==false)
%     H_final=H_0;
%     for i=1:20
%         if(i>=2)
%             W_prev=W_final;
%             H_prev=H_final;
%         else
%             W_prev=W;
%             H_prev=H_0;
%         end
%         W_final=wh*pinv(H_final);
% %         W_final=(wh*H_final')/(H*H_final');
%         W_final(W_final<0)=0;
%         H_final=pinv(W_final)*wh;
% %         H_final=(W_final'*W_final)\(W_final'*wh);
%         H_final(H_final<0)=0;
%         change_each_step(i)=norm(W_final-W_prev,'fro')/norm(W_prev,'fro')+norm(H_final-H_prev,'fro')/norm(H_prev,'fro');
%         if(change_each_step(i)<.0001)
%             i
%             break
%         end
% %         e_each_step(i)=norm((A-W_final*H_final)./sig,'fro');
%     end
%     W=W_final;
%     H=H_final;
% end
% end


    
    
            
    






    



