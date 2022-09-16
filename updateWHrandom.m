function [W,Ht,W_tilde,violation,tpg]=updateWHrandom(B,W,Ht,Q,init,alpha,beta,gamma,delta,W_tilde,like_ristretto,wtwtilde,pullW,pullH,a,b,proj_grad)
% Original Code comes from the function _update_cdnmf_fast in the Python
% library sklearn.decomposition.cdnmf_fast

% B(matrix): the lower dimension projection matrix of the data
% W(matrix): the left factor matrix
% Ht(matrix): the transpose of right factor matrix H
% Q(matrix): contains orthonormalized random samples
% init(boolean): Time efficiency: if true, updates aren't ran on code, only
% violations are calculated
% alpha(float)=L1 regularization for W, beta=L1 regularization for H
% gamma=L2 regularization for W, delta=L2 regularization for H
% W_tilde=Projected matrix W into a lower dimension
% like_ristretto(boolean)=Code difference found in the ristretto dataset: If true, before W
% is updated, QBH' is calculated to project B back into the higher
% dimensional space
% wtwtilde(boolean)=Code difference described in Erichson et al: if true,
% W'*W is calculated in the update rules for greater accuracy over the
% projected W (W_tilde'*W_tilde)

% Violation: Projected gradient in the update


tpg=0;

k=size(W,2); %Find number of factors from W
violation=0;

if(wtwtilde==false)
    WtW=W'*W; %W^T*W
else
    WtW=W_tilde'*W_tilde;
end
BtW=B'*W_tilde;
for s=1:k %s spans factors
    %t = permutation[s] %Part of Ristretto code: if rows of H are to be updated in a different order
    for i=1:size(Ht,1)
        gradHt= -BtW(i,s)+beta+delta*Ht(i,s); %gradient of (s,i) element of H
        for r=1:k
            gradHt = gradHt+WtW(s,r) * Ht(i,r); %Calculate H^T*W^T*W and add to gradient
        end
        if(pullH==true)
            gradHt=gradHt-a*(1-Ht(i,s));
        elseif(pullW==true)
            gradHt=gradHt+a*Ht(i,s);
        end

        %tic
        if(proj_grad==true)
            if(Ht(i,s) == 0) %Calculate projected gradient for H(s,i)
                pg=min(0, gradHt);
            else
                pg=gradHt;
            end
            violation = violation + pg^2; %Violation is total projected gradient
        end
        %tpg=toc;


        hess = WtW(s, s) + delta; %Hessian value
        if(pullH==true || pullW==true)
                hess=hess+a;
        end
        if(hess ~= 0)
            Ht(i, s) = max(Ht(i, s) - gradHt/hess, 0); %Update step: a rearranging of the update rule given in Erichson et al
        end
    end
end
if(like_ristretto==false) %We do not pre-project back into the higher dimension
    HHt=Ht'*Ht;
    BHt=B*Ht;
    for s=1:k
    %t = permutation[s] %Part of Ristretto code: if rows of H are to be updated in a different order
        for i=1:size(W_tilde,1)
            gradW = -BHt(i,s)+alpha+gamma*W_tilde(i,s);
            for r=1:k
                gradW = gradW+HHt(s,r) * W_tilde(i,r); 
            end
            if(pullW==true)
                gradW=gradW-b*(1-W_tilde(i,s));
            elseif(pullH==true)
                gradW=gradW+b*W_tilde(i,s);
            end


            %tic
            if(proj_grad==true)
                if(W_tilde(i,s) == 0) %Projected gradient for lower
 %                  dimension projection - consider scaling this value since
 %                  it does not reflect change in high dimension W?
                    pg=min(0, gradW);
                else
                    pg=gradW; 
                end
                violation = violation + pg^2;
            end
            %tpg=tpg+toc;



            hess = HHt(s, s) + gamma;
            if(pullH==true || pullW==true)
                hess = hess+b;
            end
            if(hess ~= 0)
                W_tilde(i, s) = W_tilde(i, s) - gradW/hess; %No max element because W_tilde=Q^T*W has no non negativity constraint
            end
        end
    end
    W=Q*W_tilde; %W is projected back into higher dimensional space
    for i=1:size(W,1)
        for j=1:k
            W(i,j)=max(0,W(i,j)); %Non negativity is now enforced
        end
    end
    W_tilde=Q'*W; %Find W_tilde by projecting back into the lower dimension
elseif(like_ristretto==true) %We project back into the higher dimension and update W
    HHt=Ht'*Ht;
    BHt=Q*B*Ht; %Project BHt into higher dimension by multiplying by Q
    for s=1:k
        %t = permutation[s]
        for i=1:size(W,1)
            gradW= -BHt(i,s)+alpha+gamma*W(i,s);
            for r=1:k
                gradW = gradW+HHt(s,r) * W(i,r); 
            end
            if(pullW==true)
                gradW=gradW-b*(1-W(i,s));
            elseif(pullH==true)
                gradW=gradW+b*W(i,s);
            end

            %tic
            if(proj_grad==true)
                if(W(i,s) == 0)
                    pg=min(0, gradW);
                else
                    pg=gradW;
                end
                violation = violation + pg^2;
            end
            %tpg=tpg+toc;


            hess = HHt(s, s) + gamma;
            if(pullW==true || pullH==true)
                hess=hess+b;
            end
            if(hess ~= 0)
                W(i, s) = max(W(i, s) - gradW/hess,0);
            end
        end
    end
    W_tilde=Q'*W;
end
end




































% % Original Code comes from the function _update_cdnmf_fast in the Python
% % library sklearn.decomposition.cdnmf_fast
% 
% % B(matrix): the lower dimension projection matrix of the data
% % W(matrix): the left factor matrix
% % Ht(matrix): the transpose of right factor matrix H
% % Q(matrix): contains orthonormalized random samples
% % sigma(boolean): if true, internally weighted NNMF is ran
% % init(boolean): Time efficiency: if true, updates aren't ran on code, only
% % violations are calculated
% % alpha(float)=L1 regularization for W, beta=L1 regularization for H
% % gamma=L2 regularization for W, delta=L2 regularization for H
% % W_tilde=Projected matrix W into a lower dimension
% % like_ristretto(boolean)=Code difference found in the ristretto dataset: If true, before W
% % is updated, QBH' is calculated to project B back into the higher
% % dimensional space
% % wtwtilde(boolean)=Code difference described in Erichson et al: if true,
% % W'*W is calculated in the update rules for greater accuracy over the
% % projected W (W_tilde'*W_tilde)
% 
% % Violation: Projected gradient in the update
% 
% 
% k=size(W,2); %Find number of factors from W
% violation=0;
% 
% 
% if(sigma==false) %If external weighting is ran
%     if(wtwtilde==false)
%         WtW=W'*W; %W^T*W
%     else
%         WtW=W_tilde'*W_tilde;
%     end
%     BtW=B'*W_tilde;
%     for s=1:k %s spans factors
%         %t = permutation[s] %Part of Ristretto code: if rows of H are to be updated in a different order
%         for i=1:size(Ht,1)
%             gradHt= -BtW(i,s)+beta+delta*Ht(i,s); %gradient of (s,i) element of H
%             for r=1:k
%                 gradHt = gradHt+WtW(s,r) * Ht(i,r); %Calculate H^T*W^T*W and add to gradient
%             end
% %             if(Ht(i,s) == 0) %Calculate projected gradient for H(s,i)
% %                 pg=min(0, gradHt);
% %             else
% %                 pg=gradHt;
% %             end
% %             violation = violation + pg^2; %Violation is total projected gradient
%             hess = WtW(s, s) + delta; %Hessian value
%             Ht(i, s) = max(Ht(i, s) - gradHt/hess, 0); %Update step: a rearranging of the update rule given in Erichson et al
%         end
%     end
% 
%     if(like_ristretto==false) %We do not pre-project back into the higher dimension
%         HHt=Ht'*Ht;
%         BHt=B*Ht;
%         for s=1:k
%             %t = permutation[s] %Part of Ristretto code: if rows of H are to be updated in a different order
%             for i=1:size(W_tilde,1)
%                 gradW = -BHt(i,s)+alpha+gamma*W_tilde(i,s);
%                 for r=1:k
%                     gradW = gradW+HHt(s,r) * W_tilde(i,r); 
%                 end
% %                if(W_tilde(i,s) == 0) %Projected gradient for lower
% %  %              dimension projection - consider scaling this value since
% %  %              it does not reflect change in high dimension W?
% %                    pg=min(0, gradW);
% %                else
% %                    pg=gradW; 
% %                end
% %                violation = violation + pg^2;
%                 hess = HHt(s, s) + gamma;
%                 %if(hess+gamma ~= 0)
%                 W_tilde(i, s) = W_tilde(i, s) - gradW/hess; %No max element because W_tilde=Q^T*W has no non negativity constraint
%                 %end
%             end
%         end
%         W=Q*W_tilde; %W is projected back into higher dimensional space
%         for i=1:size(W,1)
%             for j=1:k
%                 W(i,j)=max(0,W(i,j)); %Non negativity is now enforced
%             end
%         end
%         W_tilde=Q'*W; %Find W_tilde by projecting back into the lower dimension
%     elseif(like_ristretto==true) %We project back into the higher dimension and update W
%         HHt=Ht'*Ht;
%         BHt=Q*B*Ht; %Project BHt into higher dimension by multiplying by Q
%         for s=1:k
%             %t = permutation[s]
%             for i=1:size(W,1)
%                 gradW= -BHt(i,s)+alpha+gamma*W(i,s);
%                 for r=1:k
%                     gradW = gradW+HHt(s,r) * W(i,r); 
%                 end
% %                 if(W(i,s) == 0)
% %                     pg=min(0, gradW);
% %                 else
% %                     pg=gradW;
% %                 end
% %                 violation = violation + pg^2;
%                 hess = HHt(s, s) + gamma;
%                 %if(hess+gamma ~= 0) %%We assume this to cut down on time
%                 W(i, s) = max(W(i, s) - gradW/hess,0);
%                 %end
%             end
%         end
%         W_tilde=Q'*W;
%     end
% end


























% elseif(sigma==true && method==1)
%     for s=1:k
%         %t = permutation[s]
%         for i=1:size(Ht,1)
%             if(wtwtilde==false)
%                 modW=W(:,s).*Weights_large(:,i); %So we only have to calculate this once
%                 WtW=modW'*W;
%                 modW=W_tilde(:,s).*Weights_small(:,i);
%             elseif(wtwtilde==true)
%                 modW=W_tilde(:,s).*Weights_small(:,i);
%                 WtW=modW'*W_tilde;
%             end
%             gradHt = -B(:,i)'*modW+beta+delta*Ht(i,s);
%             for r=1:k
%                 gradHt = gradHt+WtW(r) * Ht(i,r); 
%             end
% %             if(Ht(i,s) == 0)
% %                 pg=min(0, gradHt);
% %             else
% %                 pg=gradHt(i,s);
% %             end
% %             violation = violation + pg^2;
%             hess = WtW(s);
%             %if(hess+delta ~= 0)
%             Ht(i, s) = max(Ht(i, s) - gradHt/(hess+delta), 0);
%             %end
%         end
%     end
% 
%     if(like_ristretto==false)
%         for s=1:k
%             %t = permutation[s]
%             for i=1:size(W_tilde,1)
%                 modHt=Ht(:,s).*(Weights_small(i,:)');
%                 HHt=modHt'*Ht;
%                 gradW = -B(i,:)*modHt+alpha+gamma*W_tilde(i,s);
%                 for r=1:k
%                     gradW= gradW+HHt(r) * W_tilde(i,r); 
%                 end
% % %               if(W_tilde(i,s) == 0)
% % %                     pg=min(0, gradW);
% % %               else
% %                     pg=gradW;
% % %               end
% %                 violation = violation + pg^2;
%                 hess = HHt(s);
%                 %if(hess+gamma ~= 0)
%                         W_tilde(i, s) = W_tilde(i, s) - gradW/(hess+gamma);
%                 %end
%             end
%         end
%         W=Q*W_tilde;
%         for i=1:size(W,1)
%             for j=1:k
%                 W(i,j)=max(0,W(i,j));
%             end
%         end
%         W_tilde=Q'*W;
%     elseif(like_ristretto==true)
%         for s=1:k
%             %t = permutation[s]
%             for i=1:size(W_tilde,1)
%                 modHt=Ht(:,s).*(Weights_small(i,:)');
%                 HHt=modHt'*Ht;
%                 gradW = -B(i,:)*modHt+alpha+gamma*W(i,s);
%                 for r=1:k
%                     gradW= gradW+HHt(r) * W(i,r); 
%                 end
% %                 if(W(i,s) == 0)
% %                     pg=min(0, gradW);
% %                 else
% %                     pg=gradW;
% %                 end
% %                 violation = violation + pg^2;
%                 hess = HHt(s);
%                 %if(hess+gamma ~= 0)
%                         W(i, s) = max(W(i, s) - gradW/(hess+gamma),0);
%                 %end
%             end
%         end
%         W_tilde=Q'*W;
%     end
% elseif(sigma==true && method==2)
%     for j=1:size(B,2)
%        if(shi==false)
%            Sjneg1=diag(Weights_small(:,j));
%        else
%            Sjneg1=diag(1./Sj);
%        end
%        part1=W_tilde/(W_tilde'*Sjneg1*W_tilde);
%        B1(:,j)=part1*W_tilde'*Sjneg1*B(:,j);
%     end
%     if(wtwtilde==false)
%         WtW=W'*W;
%     else
%         WtW=W_tilde'*W_tilde;
%     end
%     BtW=B1'*W_tilde;
%     for s=1:k
%         %t = permutation[s]
%         for i=1:size(Ht,1)
%             gradHt= -BtW(i,s)+beta+delta*Ht(i,s);
%             for r=1:k
%                 gradHt = gradHt+WtW(s,r) * Ht(i,r); 
%             end
% %             if(Ht(i,s) == 0)
% %                 pg=min(0, gradHt);
% %             else
% %                 pg=gradHt;
% %             end
% %             violation = violation + pg^2;
%             hess = WtW(s, s);
%             %if(hess+delta ~= 0)
%                     Ht(i, s) = max(Ht(i, s) - gradHt/(hess+delta), 0);
%             %end
%         end
%     end
% 
%     if(like_ristretto==false)
%         for i=1:size(B,1)
%             if(shi==false)
%                 Sineg1=diag(Weights_small(i,:));
%             else
%                 Sineg1=diag(1./Si);
%             end
%             part2=(Ht'*Sineg1*Ht)\Ht';
%             B1(i,:)=B(i,:)*Sineg1*Ht*part2;
%         end
%         HHt=Ht'*Ht;
%         BHt=B1*Ht;
%         for s=1:k
%             %t = permutation[s]
%             for i=1:size(W_tilde,1)
%                 gradW= -BHt(i,s)+alpha+gamma*W_tilde(i,s);
%                 for r=1:k
%                     gradW= gradW+HHt(s,r) * W_tilde(i,r); 
%                 end
% % %               if(W_tilde(i,s) == 0)
% % %                     pg=min(0, gradW);
% % %               else
% %                     pg=gradW;
% % %               end
% %                 violation = violation + pg^2;
%                 hess = HHt(s, s);
%                 %if(hess+gamma ~= 0)
%                         W_tilde(i, s) = W_tilde(i, s) - gradW/(hess+gamma);
%                 %end
%             end
%         end
%         W=Q*W_tilde;
%         for i=1:size(W,1)
%             for j=1:k
%                 W(i,j)=max(0,W(i,j));
%             end
%         end
%         W_tilde=Q'*W;
%     elseif(like_ristretto==true)
%         for i=1:size(A,1)
%             if(shi==false)
%                Sineg1=diag(Weights_small(i,:));
%             else
%                 Sineg1=diag(1./Si);
%             end
%             part2=(Ht'*Sineg1*Ht)\Ht';
%             B1(i,:)=B(i,:)*Sineg1*Ht*part2;
%         end
%         HHt=Ht'*Ht;
%         BHt=Q*B1*Ht;
%         for s=1:k
%             %t = permutation[s]
%             for i=1:size(W,1)
%                 gradW = -BHt(i,s)+alpha+gamma*W(i,s);
%                 for r=1:k
%                     gradW= gradW+HHt(s,r) * W(i,r); 
%                 end
% %                 if(W(i,s) == 0)
% %                     pg=min(0, gradW);
% %                 else
% %                     pg=gradW;
% %                 end
% %                 violation = violation + pg^2;
%                 hess = HHt(s, s);
%                 %if(hess+gamma ~= 0)
%                         W(i, s) = max(W(i, s) - gradW/(hess+gamma),0);
%                 %end
%             end
%         end
%         W_tilde=Q'*W;
%     end
% end





































% method=1;
% shi=false;
% if(sigma==true && method==1)
%     for s=1:k
%             for i=1:size(A,2)
%                 modW=W(:,s).*Weights(:,i); %So we only have to calculate this once
%                 WtW=modW'*W;
%                 gradHt(i,s)=-A(:,i)'*modW+beta+delta*Ht(i,s);
%                 for r=1:k
%                     gradHt(i,s) = gradHt(i,s)+ WtW(r)* Ht(i,r); 
%                 end                
%                 if(Ht(i,s) == 0)
%                     pg=min(0, gradHt(i,s));
%                 else
%                     pg=gradHt(i,s);
%                 end
%                 violation = violation + pg^2;
%                 %hess = W(:,s)'*modW;
%                 hess=WtW(s);
%                 if(init==false)
%                     if(hess+delta ~= 0)
%                         Ht(i, s) = max(Ht(i, s) - gradHt(i,s)/(hess+delta), 0);
%                     end     
%                 end
%             end
%     end
%         %wh=W*Ht';
%         for s=1:k
%             %t = permutation[s]
%             for i=1:size(W,1)
%                 modHt=Ht(:,s).*(Weights(i,:)');
%                 HHt=modHt'*Ht;
%                 gradW(i,s) = -A(i,:)*modHt+alpha+gamma*W(i,s);
%                 for r=1:k
%                     gradW(i,s) = gradW(i,s)+ HHt(r)* W(i,r); 
%                 end   
%                 if(W(i,s) == 0)
%                     pg=min(0, gradW(i,s));
%                 else
%                     pg=gradW(i,s);
%                 end
%                 violation = violation + pg^2;
%                 hess = HHt(s);
%                 if(init==false)
%                     if(hess+gamma ~= 0)
%                         W(i, s) = max(W(i, s) - gradW(i,s)/(hess+gamma),0);
%                     end
%                 end
%             end
%         end
% end
% if(sigma==true && method==2)
%     for j=1:size(A,2)
%        if(shi==false)
%            Sjneg1=diag(Weights(:,j));
%        else
%            Sjneg1=diag(1./Sj);
%        end
%        part1=W/(W'*Sjneg1*W);
%        A1(:,j)=part1*W'*Sjneg1*A(:,j);
%     end
%     WtW=W'*W;
%     if(random==true)
%         AtW=A1'*W_tilde; %In this case, B is used instead of the original data matrix
%     else
%         AtW=A1'*W;
%     end
%     for s=1:k
%         %t = permutation[s]
%         for i=1:size(A,2)
%             gradHt(i,s) = -AtW(i,s)+beta+delta*Ht(i,s);
%             for r=1:k
%                 gradHt(i,s) = gradHt(i,s)+WtW(s,r)*Ht(i,r);
%             end
%             if(Ht(i,s) == 0)
%                 pg=min(0, gradHt(i,s));
%             else
%                 pg=gradHt(i,s);
%             end
%             violation = violation + pg^2;
%             hess = WtW(s, s);
%             if(init==false)
%                 if(hess+delta ~= 0)
%                     Ht(i, s) = max(Ht(i, s) - gradHt(i,s)/(hess+delta), 0);
%                 end
%             end
%         end
%     end
%     for i=1:size(A,1)
%        if(shi==false)
%            Sineg1=diag(Weights(i,:));
%        else
%            Sineg1=diag(1./Si);
%        end
%        part2=(Ht'*Sineg1*Ht)\Ht';
%        A1(i,:)=A(i,:)*Sineg1*Ht*part2;
%     end
%     HHt=Ht'*Ht;
%     AHt=A1*Ht;
%     if(random==false)
%         for s=1:k
%         %t = permutation[s]
%             for i=1:size(W,1)
%                 gradW(i,s) = -AHt(i,s)+alpha+gamma*W(i,s);
%                 for r=1:k
%                     gradW(i,s) = gradW(i,s)+HHt(s,r) * W(i,r); 
%                 end
%                 if(W(i,s) == 0)
%                     pg=min(0, gradW(i,s));
%                 else
%                     pg=gradW(i,s);
%                 end
%                 violation = violation + pg^2;
%                 hess = HHt(s, s);
%                 if(init==false)
%                     if(hess+gamma ~= 0)
%                         W(i, s) = max(W(i, s) - gradW(i,s)/(hess+gamma),0);
%                     end
%                 end
%             end
%         end
%     end
% end
% if(sigma==false)
%     WtW=W'*W;
%     AtW=A'*W;
%     for s=1:k
%         %t = permutation[s]
%         for i=1:size(A,2)
%             gradHt(i,s) = -AtW(i,s)+beta+delta*Ht(i,s);
% %             checkpoint1(i,s)=gradHt(i,s);
% %             secondpart(i,s)=0;
%             for r=1:k
%                 gradHt(i,s) = gradHt(i,s)+WtW(s,r) * Ht(i,r); 
% %                 secondpart(i,s)=secondpart(i,s)+WtW(s,r) * Ht(i,r);
%             end
%             if(Ht(i,s) == 0)
%                 pg=min(0, gradHt(i,s));
%             else
%                 pg=gradHt(i,s);
%             end
%             violation = violation + pg^2;
%             hess = WtW(s, s);
%             if(init==false)
%                 if(hess+delta ~= 0)
%                     Ht(i, s) = max(Ht(i, s) - gradHt(i,s)/(hess+delta), 0);
%                 end
%             end
%         end
%     end
%     HHt=Ht'*Ht;
%     AHt=A*Ht;
%     if(random==false)
%         for s=1:k
%         %t = permutation[s]
%             for i=1:size(W,1)
%                 gradW(i,s) = -AHt(i,s)+alpha+gamma*W(i,s);
%                 for r=1:k
%                     gradW(i,s) = gradW(i,s)+HHt(s,r) * W(i,r); 
%                 end
%                 if(W(i,s) == 0)
%                     pg=min(0, gradW(i,s));
%                 else
%                     pg=gradW(i,s);
%                 end
%                 violation = violation + pg^2;
%                 hess = HHt(s, s);
%                 if(init==false)
%                     if(hess+gamma ~= 0)
%                         W(i, s) = max(W(i, s) - gradW(i,s)/(hess+gamma),0);
%                     end
%                 end
%             end
%         end
%     end
%     if(random==true)
%         for s=1:k
%             for i=1:size(W_tilde,1)
%                 gradW(i,s) = -AHt(i,s)+alpha+gamma*W_tilde(i,s);
%                 for r=1:k
%                     gradW(i,s) = gradW(i,s)+HHt(s,r) * W_tilde(i,r); 
%                 end
% %                 if(W_tilde(i,s) == 0)
% %                     pg=min(0, gradW(i,s));
% %                 else
%                     pg=gradW(i,s);
% %                 end
%                 violation = violation + pg^2;
%                 hess = HHt(s, s);
%                 if(init==false)
%                     if(hess+gamma ~= 0)
%                         W_tilde(i, s) = W_tilde(i, s) - gradW(i,s)/(hess+gamma);
%                     end
%                 end
%             end
%         end    
%     end
% end
%end