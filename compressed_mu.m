function [W,H,step,error_converge,cost_converge,adj_cost,l1cost,l2cost]=compressed_mu(A,k,p,alpha,beta,gamma,delta,tolerance,maxsteps,initialize,SigmaMat,sigma)

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
%sigma(boolean): If true, Weighted NNMF is ran, if false, external
%weighting is ran (if proj_back is also true)

% Objective: min_{W,Ht} ||(R(j)-W(:,j)*Ht(:,j)')./\SigmaMat||_F^2+
%   ||W(:,j)||_1+||Ht(:,j)||_1+||W(:,j)||_2^2+||Ht(:,j)||_2^2
%   where R(j)=A-\sum_{i ~= j}W(:,i)*Ht(:,i)'

%W,H: Factor Matrices
%Step(int): Steps in iteration
%error_converge(vector of size maxsteps+1): Weighted error at each step,
%with first element being 0th step
%cost_converge(vector of size maxsteps+1): Cost function at each step,
%with first element being 0th step
%%adj_cost(vector of size maxsteps): Cost function divided by initial value
% at each step
%l1cost(vector of size maxsteps+1): Combined L1 norm of factor matrices at
%each step, with first element being 0th step
%l2cost(vector of size maxsteps+1): Combined L2 norm of factor matrices at
%each step, with first element being 0th step

rng(range);
conv_by_cost=true; %Converge by cost function instead of projected gradient
proj_grad=false;
proj_back=true; %External weighting with post processing step


if(proj_back==true && sigma==true)  %External Weighting is ran
    A=A./SigmaMat;
end





check1=false; %First check for stopping criterion
check2=false; %Second check for stopping criterion
expectation_maximization=false;

%If matrix is column heavy
flipped=false;
[m,n]=size(A);
if(m<n)
    flipped=false;
    A=A';
    [m,n]=size(A);
end


%Initialize Factors
[W,H]=initializefactors(A,k,initialize);
Ht=H';
if(initialize=="random")
    mA=mean(mean(A));
    W=mA*W;
    Ht=mA*Ht;
end



%Randomization Step

ell=k + p; % Number of samples to take
Omega=randn(n,ell); % Random vectors
Y=A*Omega; % Random Vectors moved into the column space of data matrix
[Q,~,~]=qr(Y,0); % Orthonormalization of random vectors
% qr(Y,0) is an economy qr decomposition (if rows>cols, only first n cols 
%   of Q are find, and first n rows of R are found). If rows<=cols, qr and 
%   economy qr are same
B=Q'*A; % The projection of the data into a lower dimension is B
W_tilde=Q'*W; %Project W into a lower dimension


if(conv_by_cost==true)
    %Initialize Indicators
    cost_converge=zeros(1,maxsteps+1);
    adj_cost=zeros(1,maxsteps);
    error_converge=zeros(1,maxsteps+1);
    % violations=zeros(1,maxsteps+1);
    l1cost=zeros(1,maxsteps+1);
    l2cost=zeros(1,maxsteps+1);


    % init=true;
    % [~,~,gradW,gradHt,violation_init]=updateWH(A,W,Ht,sigma,init,alpha,beta,gamma,delta,SigmaMat,random);
    if(sigma==false)
        error_converge(1)=norm(A-W*Ht','fro')^2;
    else
        error_converge(1)=norm((A-W*Ht')./SigmaMat,'fro')^2;
    end
    W1cost=sum(sum(W));
    H1cost=sum(sum(Ht));
    W2cost=norm(W,'fro')^2;
    H2cost=norm(Ht,'fro')^2;
    l1cost(1)=alpha*W1cost+beta*H1cost;
    l2cost(1)=gamma*W2cost+delta*H2cost;
    cost_converge(1)=error_converge(1)+l1cost(1)+l2cost(1);
    if(alpha==0 && beta==0)
        l1cost(1)=W1cost+H1cost;
    end
    if(gamma==0 && delta==0)
        l2cost(1)=W2cost+H2cost;
    end
end

%Calculate Weights
if(sigma==true)
    Weights=ones(size(A))./(SigmaMat.^2);
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




%Main algorithm
init=false;
for step=1:maxsteps

    violation=0;
    hgradp=(Ht*(W_tilde'*W_tilde)+delta*Ht+beta*ones(size(Ht)));
    hgradn=(B'*W_tilde);
    hgrad=hgradp-hgradn;
    Ht=Ht.*(hgradn./hgradp);
    %Ht(Ht<0)=0;
    wgradp=W_tilde*(Ht'*Ht)+gamma*W_tilde+alpha*ones(size(W_tilde));
    wgradn=B*Ht;
    wgrad=wgradp-wgradn;
    W_tilde=W_tilde.*(wgradn./wgradp);
    W=Q*W_tilde;
    %W(W<0)=.001;
    W_tilde=Q'*W;
    if(proj_grad==true)
        for p=1:size(W,2)
            for i=1:size(W,1)
                if(W(i,p)==0)
                    pg=min(0,wgrad(i,p));
                else
                    pg=wgrad(i,p);
                end
                violation=violation+pg^2;
            end
            for j=1:size(H,2)
                if(Ht(j,p)==0)
                    pg=min(0,hgrad(j,p));
                else
                    pg=hgrad(j,p);
                end
                violation=violation+pg^2;
            end
        end
    end
    if(conv_by_cost==true)
        if(sigma==false)
            error_converge(step+1)=norm(A-W*Ht','fro')^2;
        else
            error_converge(step+1)=norm((A-W*Ht')./SigmaMat,'fro')^2;
        end
        W1cost=sum(sum(W));
        H1cost=sum(sum(Ht));
        W2cost=norm(W,'fro')^2;
        H2cost=norm(Ht,'fro')^2;
        l1cost(step+1)=alpha*W1cost+beta*H1cost;
        l2cost(step+1)=gamma*W2cost+delta*H2cost;
        cost_converge(step+1)=error_converge(step+1)+l1cost(step+1)+l2cost(step+1);
        adj_cost(step+1)=cost_converge(step+1)/cost_converge(1);
        if(alpha==0 && beta==0)
            l1cost(step+1)=W1cost+H1cost;
        end
        if(gamma==0 && delta==0)
            l2cost(step+1)=W2cost+H2cost;
        end
        %     violations(step+1)=violation;
        stop_cond=abs(adj_cost(step+1)-adj_cost(step));
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
        if(stop_cond<=tolerance) %If chg(J)/J_0 <= tolerance
            if(check1==true) %If one time before, this was true
                if(check2==true) %If two times before, this was true
                    break
                else
                    check2=true;
                end
            else
                check1=true;
            end
        else
            check1=false; %Reset checks
            check2=false; %Reset checks
        end
    end
    
   
end
step
W(W<0)=0; %Non zero elements are set positive after iteration (to avoid 
% setting to zero in iteration)
Ht(Ht<0)=0;
W1=W;
H=Ht';





if(proj_back==true && sigma==true) %External Weighting is ran
    change_each_step=zeros(1,20); % Change in factor matrices throughout the
%    iterations
    wh=(W*H).*SigmaMat; %Actual factors W*H is related to calculated factor matrices by (W_est*H_est).*SigmaMat
    H_final=sqrt(mean(A,'all')/k)*(H/mean(H,'all')); % Initial H in iteration is 
%    estimated as the values of H scaled to multiply by W to equal the mean of A
    for i=1:20 %Iteration step
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
        if(change_each_step(i)<.0001)
            disp(i) %Print step if convergence is reached
            break %Break out of for loop
        end
%         e_each_step(i)=norm((A-W_final*H_final)./sig,'fro');
    end
    W=W_final; %New W is W final
    H=H_final;
end

if(flipped==true)
    W1=W; %Store values of W as W1
    W=Ht; %Set W equal to transpose of H
    H=(W1)'; %Set H equal to transpose of W
end




if(flipped==true)
    W=Ht;
    H=(W1)';
end
end

