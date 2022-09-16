%%Matlab Code from https://www.researchgate.net/profile/N-Erichson/publication/320891179_Randomized_Nonnegative_Matrix_Factorization/links/5af889980f7e9b026beb41ec/Randomized-Nonnegative-Matrix-Factorization.pdf?origin=publication_detail
function [W,H,step,error_converge,cost_converge,adj_cost,l1cost,l2cost]=exactnnmf(A,k,alpha,beta,gamma,delta,tolerance,maxsteps,initialize,SigmaMat,sigma,pullW,pullH,a,b,range)
rng(range);
conv_by_cost=false;
proj_grad=true;
proj_back=true;
rank_one_scaling=false;


%Deal with column heavy matrix
flipped=false;
[m,n]=size(A);
if(m<n)
    flipped=false;
    A=A';
    [m,n]=size(A);
end


if(rank_one_scaling==true && sigma==false)
    Weights=ones(size(A))./SigmaMat;
    B=ones(1,m);
    C=zeros(n,1);
    for its=1:2
        for j=1:n
            C(j)=B*Weights(:,j)/(B*B');
        end
        for i=1:m
            B(i)=Weights(i,:)*C/(C'*C);
        end
    end
    A1=A;
    A=repmat(B',1,n).*A.*repmat(C',m,1);
end



if(proj_back==true && sigma==false)  %External Weighting is ran
    A1=A;
    A=A./SigmaMat;
end


check1=false; %First check for stopping criterion
check2=false; %Second check for stopping criterion
%This is translated from Python function compute_rnmf from Ristretto Folder
%Folder linked from Erichson '18

%A: matrix, k: target rank, gamma: l2 regularization for
%W, delta: l2 regularization for H, alpha: l1 regularization for W, beta:
%l1 regularization for H, tolerance: for stopping algorithm, maxsteps:
%maximum steps of the algorithm

expectation_maximization=false;

%Deal with this when making final algorithm
if(~exist('maxsteps','var'))
    maxsteps=100;
end
if(~exist('tolerance','var'))
    tolerance=1*10^(-3);
end
if(~exist('initialize','var'))
    initialize='random';
end

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





%Expectation Maximum Calculation
if(expectation_maximization==true)
    weights1=ones(size(A))./SigmaMat;
end

%Set weights
Weights=ones(size(A));
if(sigma==true && expectation_maximization==false)
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


%     Wnorm=norm(W,'fro');
%     Hnorm=norm(H,'fro');
%     gradWnorm=norm(gradW,'fro');
%     gradHnorm=norm(gradHt,'fro');
%     numzerosW=length(W(W==0));
%     numzerosH=length(Ht(Ht==0));
%     data=[step,violations(step),error_converge(step),weighted_errorconverge(step),Wvalues(1,step),Wvalues(2,step),Wvalues(3,step),Wvalues(4,step),Hvalues(1,step),Hvalues(2,step),Hvalues(3,step),Hvalues(4,step)];
%     fprintf('%d\t\t\t%.2f\t\t\t%.5f\t\t%.5f\t\t\t\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n',data)
%     data=[step,violations(step),error_converge(step),violationW,violationH,gradWnorm,gradHnorm,dgW,dgH];
%     fprintf('%d\t\t\t%.2f\t\t\t%.5f\t\t%.2f\t\t%.2f\t\t\t\t%.2f\t\t\t%.2f\t\t\t\t%d\t\t%d\n',data)
%     data=[step,violations(step),error_converge(step),weighted_errorconverge(step)];
%     fprintf('%d\t\t\t%.2f\t\t\t%.5f\t\t%.5f\n',data)
%     if(step>=3)
%         if(std([violations(step),violations(step-1),violations(step-2)])/mean(violations(2:step))<=tolerance || (violations(step)>violations(step-1)&violations(step-1)>violations(step-2)))
%             break
%         end
%     end

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






%     if(proj_back==true && sigma==false)
%         H_exact=Ht';
%         W_exact=W;
%         H_0=sqrt(mean(A.*SigmaMat,'all')/k)*(H_exact/mean(H_exact,'all'));
%         change_each_step=zeros(1,20);
%         wh=(W_exact*H_exact).*SigmaMat;
%         if(sigma==false)
%     %H_final=sqrt(mean(A_small,'all')/k)*(0.5*ones(size(H_exact))+rand(size(H_exact))); %H is initiated near the values of the actual H
%             H_final=H_0;
%             for i=1:20
%                 if(i>=2)
%                     W_prev=W_final;
%                     H_prev=H_final;
%                 else
%                     W_prev=W_exact;
%                     H_prev=H_0;
%                 end
%                 W_final=wh*pinv(H_final);
%                 H_final=pinv(W_final)*wh;
%                 change_each_step(i)=norm(W_final-W_prev,'fro')/norm(W_prev,'fro')+norm(H_final-H_prev,'fro')/norm(H_prev,'fro');
%                 if(change_each_step(i)<.0001)
%                     i
%                     break
%                 end
% %         e_each_step(i)=norm((A-W_final*H_final)./sig,'fro');
%             end
%             W_last=W;
%             H_last=Ht';
%             W=W_final;
%             Ht=H_final';
%         end
%         error_converge(step+1)=norm((A.*SigmaMat-W*Ht')./SigmaMat,'fro')^2;
%         W1cost=sum(sum(W));
%         H1cost=sum(sum(Ht));
%         W2cost=norm(W,'fro')^2;
%         H2cost=norm(Ht,'fro')^2;
%         l1cost(step+1)=alpha*W1cost+beta*H1cost;
%         l2cost(step+1)=gamma*W2cost+delta*H2cost;
%         cost_converge(step+1)=error_converge(step+1)+l1cost(step+1)+l2cost(step+1);
%         adj_cost(step+1)=cost_converge(step+1)/cost_converge(1);
%         if(alpha==0 && beta==0)
%             l1cost(step+1)=W1cost+H1cost;
%         end
%         if(gamma==0 && delta==0)
%             l2cost(step+1)=W2cost+H2cost;
%         end
%         Ht=H_last';
%         W=W_last;
%     end 








% % init=true;
% % [~,~,violation_init]=updateWH(A,W,Ht,sigma,init,alpha,beta,gamma,delta,SigmaMat,random);
% step=0;
%     if(proj_back==false || sigma==true)
%         if(sigma==false)
%             error_converge(step+1)=norm(A-W*Ht','fro')^2;
%         else
%             error_converge(step+1)=norm((A-W*Ht')./SigmaMat,'fro')^2;
%         end
%         W1cost=sum(sum(W));
%         H1cost=sum(sum(Ht));
%         W2cost=norm(W,'fro')^2;
%         H2cost=norm(Ht,'fro')^2;
%         l1cost(step+1)=alpha*W1cost+beta*H1cost;
%         l2cost(step+1)=gamma*W2cost+delta*H2cost;
%         cost_converge(step+1)=error_converge(step+1)+l1cost(step+1)+l2cost(step+1);
%         adj_cost(step+1)=cost_converge(step+1)/cost_converge(1);
%         if(alpha==0 && beta==0)
%             l1cost(step+1)=W1cost+H1cost;
%         end
%         if(gamma==0 && delta==0)
%             l2cost(step+1)=W2cost+H2cost;
%         end
%     end
%     
%     
%     
%     
%     
%     if(proj_back==true && sigma==false)
%         H_exact=Ht';
%         W_exact=W;
%         H_0=sqrt(mean((A.*SigmaMat),'all')/k)*(H_exact/mean(H_exact,'all'));
%         change_each_step=zeros(1,20);
%         wh=(W_exact*H_exact).*SigmaMat;
%         if(sigma==false)
%     %H_final=sqrt(mean(A_small,'all')/k)*(0.5*ones(size(H_exact))+rand(size(H_exact))); %H is initiated near the values of the actual H
%             H_final=H_0;
%             for i=1:20
%                 if(i>=2)
%                     W_prev=W_final;
%                     H_prev=H_final;
%                 else
%                     W_prev=W_exact;
%                     H_prev=H_0;
%                 end
%                 W_final=wh*pinv(H_final);
%                 H_final=pinv(W_final)*wh;
%                 change_each_step(i)=norm(W_final-W_prev,'fro')/norm(W_prev,'fro')+norm(H_final-H_prev,'fro')/norm(H_prev,'fro');
%                 if(change_each_step(i)<.0001)
%                     i
%                     break
%                 end
% %         e_each_step(i)=norm((A-W_final*H_final)./sig,'fro');
%             end
%             W_last=W;
%             H_last=Ht';
%             W=W_final;
%             Ht=H_final';
%         end
%         error_converge(step+1)=norm((A.*SigmaMat-W*Ht')./SigmaMat,'fro')^2;
%         W1cost=sum(sum(W));
%         H1cost=sum(sum(Ht));
%         W2cost=norm(W,'fro')^2;
%         H2cost=norm(Ht,'fro')^2;
%         l1cost(step+1)=alpha*W1cost+beta*H1cost;
%         l2cost(step+1)=gamma*W2cost+delta*H2cost;
%         cost_converge(step+1)=error_converge(step+1)+l1cost(step+1)+l2cost(step+1);
%         adj_cost(step+1)=cost_converge(step+1)/cost_converge(1);
%         if(alpha==0 && beta==0)
%             l1cost(step+1)=W1cost+H1cost;
%         end
%         if(gamma==0 && delta==0)
%             l2cost(step+1)=W2cost+H2cost;
%         end
%         Ht=H_last';
%         W=W_last;
%     end 
