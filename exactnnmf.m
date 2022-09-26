%%Matlab Code from https://www.researchgate.net/profile/N-Erichson/publication/320891179_Randomized_Nonnegative_Matrix_Factorization/links/5af889980f7e9b026beb41ec/Randomized-Nonnegative-Matrix-Factorization.pdf?origin=publication_detail
function [W,H,step,error_converge,cost_converge,adj_cost,l1cost,l2cost]=exactnnmf(A,k,alpha,beta,gamma,delta,tolerance,maxsteps,initialize,SigmaMat,sigma,pullW,pullH,a,b)


%This is translated from Python function compute_rnmf from Ristretto Folder
%Folder referred to from Erichson '17

%A(matrix): Data Matrix
%k(int): Target rank
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
%violations(vector of size maxsteps+1): -- sub out for error_converge in 
% return values if proj_grad is true -- Violation (Projected Gradient 
%   Squared) at each step, with first element being 0th step
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





conv_by_cost=false; %Converge by cost function
proj_grad=true; %Projected gradien
proj_back=true; %Project back using external weighting


%Deal with column heavy matrix
flipped=false;
[m,n]=size(A);
if(m<n)
    flipped=false;
    A=A';
    [m,n]=size(A);
end



if(proj_back==true && sigma==false)  %External Weighting is ran
    A1=A;
    A=A./SigmaMat;
end


check1=false; %First check for stopping criterion
check2=false; %Second check for stopping criterion


%Initialization
[W,H]=initializefactors(A,k,initialize);
%W=sqrt(1.9969*10^(-5)/k)*(W/mean(W,'all'));
Ht=H';
if(initialize=="random")
    mA=mean(mean(A));
    W=mA/sqrt(k)*W;
    Ht=mA/sqrt(k)*Ht;
end

if(conv_by_cost==true)
    %Initialize Indicators
    cost_converge=zeros(1,maxsteps+1);
    adj_cost=zeros(1,maxsteps);
    error_converge=zeros(1,maxsteps+1);
    % violations=zeros(1,maxsteps+1);
    l1cost=zeros(1,maxsteps+1);
    l2cost=zeros(1,maxsteps+1);
end



%First Error Calculation
if(conv_by_cost==true)
    if(sigma==false)
        error_converge(1)=norm(A-W*Ht','fro')^2;
    else
        error_converge(1)=norm((A-W*Ht')./SigmaMat,'fro')^2;
    end

    %Initial Indicator Calculations
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


%Set weights
Weights=ones(size(A));
if(sigma==true)
    Smat2=SigmaMat.^2;
    Weights=Weights./Smat2;
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





%Main Steps
init=false;
for step=1:maxsteps
    
    %Compress A via Expectation Maximization Technique, otherwise run step
    if(expectation_maximization==true)
       sigma=false;
       Acomp=weights1.*A+(ones(size(A))-weights1).*(W*Ht');
       [W,Ht,violation]=updateWH(Acomp,W,Ht,sigma,init,alpha,beta,gamma,delta,Weights,phi,fpeak);
    else
       [W,Ht,violation]=updateWH_pull(A,W,Ht,sigma,alpha,beta,gamma,delta,Weights,pullW,pullH,a,b,proj_grad);
    end
    
    
    

    % Violation Checks
%     if(violation==0)
%             %disp("error")
%     end
%     if(step==1)
%         violation_init=violation;
%     end

%     if(violation/violation_init<=tolerance)
%         break
%     end








    % Calculate Residual, Weighted Error, Cost
%     reg_error(step+1)=norm(A-W*Ht','fro')^2;
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
        adj_violation(step+1)=violations(step)/violations(1);
        if(step>=2)
            stop_cond=abs(adj_violation(step)-adj_violation(step-1));
        end
        %time_projgrad(step+1)=time_projgrad(step+1)+toc;
    end

    
    %     violations(step+1)=violation;
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


if(rank_one_scaling==true && sigma==false)
    invB=ones(size(B'))./B';
    W=repmat(invB,1,k).*W;
    invC=ones(size(C'))./C';
    H=H.*repmat(invC,k,1);
end


if(proj_back==true && sigma==false) %Weighted NNMF is ran


    iter=20;
    tol_pb=10^(-4);
    change_each_step=zeros(1,iter); % Change in factor matrices throughout the
    %    iterations
    %e_each_step=zeros(1,iter);
    wh=(W*H).*SigmaMat; %Actual factors W*H is related to calculated factor matrices by (W_est*H_est).*SigmaMat
    H_final=sqrt(mean(A1,'all')/k)*(H/mean(H,'all')); % Initial H in iteration is 
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

