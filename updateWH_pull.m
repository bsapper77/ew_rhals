function [W,Ht,violation]=updateWH_pull(A,W,Ht,sigma,alpha,beta,gamma,delta,Weights,pullW,pullH,a,b,proj_grad)
% Original Code comes from the function _update_cdnmf_fast in the Python
% library sklearn.decomposition.cdnmf_fast

% A(matrix): data matrix
% W(matrix): the left factor matrix
% Ht(matrix): the transpose of right factor matrix H
% alpha(float)=L1 regularization for W, beta=L1 regularization for H
% gamma(float)=L2 regularization for W, delta=L2 regularization for H
% Weights(matrix)= Matrix of 1./SigmaMat^2 (element-wise division)

% pullW(boolean): If true, W is pulled up, pullH is false
% pullH(boolean): If true, H is pulled up, pullW is false.
% If either is true, the other factor matrix is pulled down
% a(double): pulling of H
% b(double): pulling of W
% proj_grad(boolean): Projected gradient used as stopping condition

% W: Left factor matrix
% Ht: Transpose of H; right factor matrix
% Violation: Projected gradient in the update


k=size(W,2);



violation=0;

if(sigma==true) %Weighting within step
    for s=1:k
        for i=1:size(A,2)
            modW=W(:,s).*Weights(:,i); %So we only have to calculate this once
            WtW=modW'*W;
            gradHt=-A(:,i)'*modW+beta+delta*Ht(i,s);
            for r=1:k
                gradHt = gradHt+ WtW(r)* Ht(i,r);
            end
            if(proj_grad==true)
                if(Ht(i,s) == 0)
                    pg=min(0, gradHt);
                else
                    pg=gradHt;
                end
                violation = violation + pg^2;
            end
            hess=WtW(s)+delta;
            %if(hess+delta ~= 0)
            Ht(i, s) = max(Ht(i, s) - gradHt/(hess), 0);
            %Ht(i,s)=max((Ht(i,s)*hess-gradHt)/(hess+delta),0);
            %end
        end

    end
    for s=1:k
        for i=1:size(W,1)
            modHt=Ht(:,s).*(Weights(i,:)');
            HHt=modHt'*Ht;
            gradW = -A(i,:)*modHt+alpha+gamma*W(i,s);
            for r=1:k
                gradW= gradW+ HHt(r)* W(i,r);
            end
            if(proj_grad==true)
                if(W(i,s) == 0)
                    pg=min(0, gradW);
                else
                    pg=gradW;
                end
                violation = violation + pg^2;
            end
            hess = HHt(s)+gamma;
            %if(hess+gamma ~= 0)
            W(i, s) = max(W(i, s) - gradW/(hess),0);
            %W(i,s)=max((hess*W(i,s)-gradW)/(hess+gamma),0);
            %end
        end
    end
end
if(sigma==false) %No weighting within step
    WtW=W'*W;
    AtW=A'*W;
    for s=1:k
        for i=1:size(A,2)
            gradHt = -AtW(i,s)+beta+delta*Ht(i,s);

            for r=1:k
                gradHt= gradHt+WtW(s,r) * Ht(i,r); 
            end
            if(pullH==true)
                gradHt=gradHt-a*(1-Ht(i,s));
            elseif(pullW==true)
                gradHt=gradHt+a*Ht(i,s);
            end
            if(proj_grad==true)
                if(Ht(i,s) == 0)
                    pg=min(0, gradHt);
                else
                    pg=gradHt;
                end
                violation = violation + pg^2;
            end
            hess = WtW(s,s)+delta;
            if(pullW==true || pullH==true)
                hess=hess+a;
            end
            %if(hess+delta ~= 0)
                    Ht(i, s) = max(Ht(i, s) - gradHt/(hess), 0);
                    %Ht(i,s)=max((Ht(i,s)*hess-gradHt)/(hess+delta),0);
            %end
        end
    end
    HHt=Ht'*Ht;
    AHt=A*Ht;
    for s=1:k
        for i=1:size(W,1)
            gradW = -AHt(i,s)+alpha+gamma*W(i,s);
            for r=1:k
                gradW= gradW+HHt(s,r) * W(i,r); 
            end
            if(pullW==true)
                gradW=gradW-b*(1-W(i,s));
            elseif(pullH==true)
                gradW=gradW+b*W(i,s);
            end
            if(proj_grad==true)
                if(W(i,s) == 0)
                    pg=min(0, gradW);
                else
                    pg=gradW;
                end
                violation = violation + pg^2;
            end
            hess = HHt(s,s)+gamma;
            if(pullW==true || pullH==true)
                hess=hess+b;
            end            
%             end
            %if(hess+gamma ~= 0)
                  W(i, s) = max(W(i, s) - gradW/(hess),0);
%                 W(i,s)=max((hess*W(i,s)-gradW)/(hess+gamma),0);
            %end
        end
    end
end
end