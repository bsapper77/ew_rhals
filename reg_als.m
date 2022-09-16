function [W,H,step,error_converge,cost_converge,adj_cost,l1cost,l2cost]=reg_als(A,k,alpha,beta,gamma,delta,tolerance,maxsteps,initialize,SigmaMat,sigma,range)
rng(range)
conv_by_cost=true;
proj_grad=false;
proj_back=true;


if(proj_back==true && sigma==false)  %External Weighting is ran
    A1=A;
    A=A./SigmaMat;
end



A1=zeros(size(A));
pmf=true;
check1=false; %First check for stopping criterion
check2=false; %Second check for stopping criterion
method=1;
shi=false;
%This is translated from Python function compute_rnmf from Ristretto Folder
%Folder linked from Erichson '18

%A: matrix, k: target rank, gamma: l2 regularization for
%W, delta: l2 regularization for H, alpha: l1 regularization for W, beta:
%l1 regularization for H, tolerance: for stopping algorithm, maxsteps:
%maximum steps of the algorithm

expectation_maximization=false;

%If matrix is column heavy
flipped=false;
[m,n]=size(A);
if(m<n)
    flipped=false;
    A=A';
    [m,n]=size(A);
end

%Add this later?
if(~exist('maxsteps','var'))
    maxsteps=100;
end
if(~exist('tolerance','var'))
    tolerance=1*10^(-3);
end
if(~exist('initialize','var'))
    initialize='random';
end

%Initialize
[W,H]=initializefactors(A,k,initialize);
Ht=H';
if(initialize=="random")
    mA=mean(mean(A));
    W=mA*W;
    Ht=mA*Ht;
end

if(conv_by_cost==true)
    %Initialize Indicators
    cost_converge=zeros(1,maxsteps+1);
    adj_cost=zeros(1,maxsteps);
    error_converge=zeros(1,maxsteps+1);
    l1cost=zeros(1,maxsteps+1);
    l2cost=zeros(1,maxsteps+1);
end

if(conv_by_cost==true)
    %First Indicator values
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
%Expectation Maximization Step
if(expectation_maximization==true)
    weights1=ones(size(A))./SigmaMat;
end

%Set Weights
Weights=ones(size(A));
if(sigma==true && expectation_maximization==false)
    if(method==1 || shi==false)
        Smat2=SigmaMat.^2;
        Weights=Weights./Smat2;
    end
    if(shi==true && method==2)
        Si=sum(Smat2,1);
        Sj=sum(Smat2,2);
    end
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
if(expectation_maximization==true)
    sigma=false;
    true_A=A; %Remember what original data matrix is
end
for step=1:maxsteps
    %Alternating Least Squares Internal Weighting
    if(expectation_maximization==true)
        A=weights1.*A+(ones(size(A))-weights1).*(W*H);
    end
    if(sigma==true && method==1)
        violation=0;
        for j=1:size(A,2)
            H1=W'*(repmat(Weights(:,j),1,k).*W)+delta*eye(k);
            H2=W'*(A(:,j).*Weights(:,j))-beta*ones(k,1);
            if(proj_grad==true)
                grad=H1*H(:,j)-H2;
                for s=1:size(H,1)
                    if(H(s,j) == 0)
                        pg=min(0, grad(s));
              	    else
                 	    pg=grad(s);
                    end
                    violation = violation + pg^2;
                end
            end
            H(:,j)=H1\H2;
            for i=1:size(H,1)
                if(H(i,j)<0)
                    H(i,j)=0;
                end
            end
        end
        for i=1:size(A,1)
            W1=(Weights(i,:).*A(i,:))*H'-alpha*ones(1,k);
            W2=(H.*repmat(Weights(i,:),k,1))*H'+gamma*eye(k);
            if(proj_grad==true)
                grad=-W1+W(i,:)*W2;
                for s=1:size(W,2)
                    if(H(s,j) == 0)
                        pg=min(0, grad(s));
              	    else
                 	    pg=grad(s);
                    end
                    violation = violation + pg^2;
                end
            end
            W(i,:)=W1/W2;
            if(pmf==true)
                W(W<0)=0;
            end
        end
    end
    if(sigma==true && method==2)
       for j=1:size(A,2)
           if(shi==false)
               Sjneg1=diag(Weights(:,j));
           else
               Sjneg1=diag(1./Sj);
           end
       part1=W/(W'*Sjneg1*W);
       A1(:,j)=part1*W'*Sjneg1*A(:,j);
       end
       H1=W'*W+delta*eye(k);
       for j=1:size(A,2)
           H2=W'*A1(:,j)-beta*ones(k,1);
           H(:,j)=H1\H2;
       end
       if(pmf==true)
           H(H<0)=0;
       end
       for i=1:size(A,1)
           if(shi==false)
               Sineg1=diag(Weights(i,:));
           else
               Sineg1=diag(1./Si);
           end
           part2=(Ht'*Sineg1*Ht)\Ht';
           A1(i,:)=A(i,:)*Sineg1*Ht*part2;
       end
       W2=H*H'+gamma*eye(k);
       for i=1:size(A,1)
           W1=A1(i,:)*H'-alpha*ones(1,k);
           W(i,:)=W1/W2;
       end
       if(pmf==true)
           W(W<0)=0;
       end
    end
    if(sigma==false)
        violation=0;
        H1=W'*W+delta*eye(k);
        for j=1:size(A,2)
            H2=W'*A(:,j)-beta*ones(k,1);
            if(proj_grad==true)
                grad=H1*H(:,j)-H2;
                for s=1:size(H,1)
                    if(H(s,j) == 0)
                        pg=min(0, grad(s));
              	    else
                 	    pg=grad(s);
                    end
                    violation = violation + pg^2;
                end
            end
            H(:,j)=H1\H2;
            for i=1:size(H,1)
                if(H(i,j)<0)
                    H(i,j)=0;
                end
            end
        end
        W2=H*H'+gamma*eye(k);
        for i=1:size(A,1)
            W1=A(i,:)*H'-alpha*ones(1,k);
            if(proj_grad==true)
                grad=-W1+W(i,:)*W2;
                for s=1:size(W,2)
                    if(H(s,j) == 0)
                        pg=min(0, grad(s));
              	    else
                 	    pg=grad(s);
                    end
                    violation = violation + pg^2;
                end
            end
            W(i,:)=W1/W2;
            for j=1:size(W,2)
                if(W(i,j)<0)
                    W(i,j)=0;
                end
            end
        end
    end
    Ht=H';
    
    %Indicator Calculation
    if(expectation_maximization==true)
        A=true_A; %remember what true A is
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
H=Ht';







if(proj_back==true && sigma==false) %External Weighting is ran
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

%Initialize With ones
% fac=rand(1,1);
% if(sigma==false)
%     W=ones(m,k)/sqrt(k);
%     Ht=ones(n,k)/sqrt(k);
% else
%     W=B;
%     Ht=C';
% end


% if(~exist('SigmaMat','var'))
%     sigma=false;
%     Weights=ones(m,n);
% else
%     Weights=ones(m,n)./SigmaMat;
%     %sigma=true;
% end
