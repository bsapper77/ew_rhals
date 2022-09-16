function [W,Ht,violation]=updateWH(A,W,Ht,sigma,alpha,beta,gamma,delta,Weights,pullW,pullH,a,b,proj_grad)
A1=zeros(size(A));

k=size(W,2);

% fpeak=true;
fpeak=false;
phi=0;
% Phi=(0)*ones(k);
Phi=phi*ones(k);
Phi=Phi-diag(diag(Phi));
softness=10^(-3);
if(fpeak==true)
    D=(W/(W'*W))*Phi;
    D_h=(((-1)*Phi)/(Ht'*Ht))*Ht';
end
    

% pullH=true;
% pullW=false;


violation=0;
% gradW=zeros(size(A,1),k); %initialize G gradient
% gradHt=zeros(size(A,2),k); %initialize H gradient
% if(~exist(Weights,'var'))
%     Weights=ones(size(A,1),size(A,2));
% end
method=1;
shi=false;
if(sigma==true && method==2)
    Si=sum(Smat2,1);
    Sj=sum(Smat2,2);
end
if(sigma==true && method==1)
%     mht=max(Ht,[],'all')
    for s=1:k
        if(pullH==true)
            pullH_l1=abs(length(Ht(:,s))-sum(Ht(:,s))); %||1-H(j,:)||_1
        end
        if(pullW==true)
            pullH_l1=sum(Ht(:,s)); %||H(j,:)||_1
        end
        for i=1:size(A,2)
%             if(step==1)
%                 W(:,s)
%                 W(:,s).*Weights(:,i)
%                 (W(:,s).*Weights(:,i))'*W
%             end
            modW=W(:,s).*Weights(:,i); %So we only have to calculate this once
            WtW=modW'*W;
        	gradHt=-A(:,i)'*modW+beta+delta*Ht(i,s);
        	for r=1:k
                gradHt = gradHt+ WtW(r)* Ht(i,r); 
            end
            if(pullH==true)
                gradHt=gradHt-a*pullH_l1; %a is 1/a^2 from paper
            elseif(pullW==true)
                gradHt=gradHt+a*pullH_l1; %a is 1/a^2 from paper
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
            if(pullH==true || pullW==true)
                hess=hess+a;
            end
            %if(hess+delta ~= 0)
                Ht(i, s) = max(Ht(i, s) - gradHt/(hess), 0);
                %Ht(i,s)=max((Ht(i,s)*hess-gradHt)/(hess+delta),0);
            %end     
        end
    end
     if(fpeak==true)
         Pull=W+D;
     end
%      mw=max(W,[],'all');
        %wh=W*Ht';
        for s=1:k
            if(pullH==true)
                pullW_l1=sum(W(:,s)); %||W(:,s)||_1
            end
            if(pullW==true)
                pullW_l1=abs(length(W(:,s))-sum(W(:,s))); %||1-W(:,s)||_1
            end
            %t = permutation[s]
            for i=1:size(W,1)
                modHt=Ht(:,s).*(Weights(i,:)');
                HHt=modHt'*Ht;
                gradW = -A(i,:)*modHt+alpha+gamma*W(i,s);
                if(fpeak==true)
                    gradW=gradW-(Pull(i,s)-W(i,s))/softness^2;
                end
                if(pullW==true)
                    gradW=gradW-b*pullW_l1;
                elseif(pullH==true)
                    gradW=gradW+b*pullW_l1;
                end
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
                if(fpeak==true)
                    hess = hess+1/softness^2;
                end
                if(pullH==true || pullW==true)
                    hess = hess+b;
                end
                %if(hess+gamma ~= 0)
                    W(i, s) = max(W(i, s) - gradW/(hess),0);
                    %W(i,s)=max((hess*W(i,s)-gradW)/(hess+gamma),0);
                %end
            end
        end
end
if(sigma==false)
    WtW=W'*W;
    AtW=A'*W;
    maxh=max(Ht,[],'all');
    if(fpeak==true)
        Pull=Ht+D_h';
    end
    for s=1:k
        if(pullH==true)
            pullH_l1=abs(length(Ht(:,s))-sum(Ht(:,s))); %||1-H(j,:)||_1
            %pullH_l1=abs(maxh*length(Ht(:,s))-sum(Ht(:,s))); %||max(H)-H(j,:)||_1
        end
        if(pullW==true)
            pullH_l1=sum(Ht(:,s)); %||H(j,:)||_1
        end
%         if(fpeak==true)
%             %D=(W/(W'*W))*Phi;
%             D_h=(((-1)*Phi)/(Ht'*Ht))*Ht';
%             Pull=Ht+D_h';
%         end
        %t = permutation[s]
        for i=1:size(A,2)
            gradHt = -AtW(i,s)+beta+delta*Ht(i,s);
%             if(fpeak==true)
%                 gradHt=gradHt-(Pull(i,s)-Ht(i,s))/softness^2;
%             end
            for r=1:k
                gradHt= gradHt+WtW(s,r) * Ht(i,r); 
            end
            if(pullH==true)
                gradHt=gradHt-a*pullH_l1; %a is 1/a^2 from paper
            elseif(pullW==true)
                gradHt=gradHt+a*pullH_l1; %a is 1/a^2 from paper
            end
            if(proj_grad==true)
                if(Ht(i,s) == 0)
                    pg=min(0, gradHt);
                else
                    pg=gradHt;
                end
                violation = violation + pg^2;
            end
%             if(fpeak==true)
%                 hess = WtW(s,s)+delta+1/softness^2;
%             else
            hess = WtW(s,s)+delta;
%             end
            if(pullH==true || pullW==true)
                hess=hess+a;
            end
%             hess = WtW(s, s);
            %if(hess+delta ~= 0)
                    Ht(i, s) = max(Ht(i, s) - gradHt/(hess), 0);
                    %Ht(i,s)=max((Ht(i,s)*hess-gradHt)/(hess+delta),0);
            %end
        end
    end
    HHt=Ht'*Ht;
    AHt=A*Ht;
    maxw=max(W,[],'all');
%     if(fpeak==true)
%         Pull=W+D;
%     end
    for s=1:k
%         if(fpeak==true)
%             D=(W/(W'*W))*Phi;
% %           D_h=(((-1)*Phi)/(Ht'*Ht))*Ht';
%             Pull=W+D;
%         end
        if(pullH==true)
            pullW_l1=sum(W(:,s)); %||W(:,s)||_1
        end
        if(pullW==true)
            pullW_l1=abs(length(W(:,s))-sum(W(:,s))); %||1-W(:,s)||_1
            %pullW_l1=abs(maxw*length(W(:,s))-sum(W(:,s))); %||max(W)-W(:,s)||_1
        end
    %t = permutation[s]
        for i=1:size(W,1)
            gradW = -AHt(i,s)+alpha+gamma*W(i,s);
            %if(fpeak==true)
            %    gradW=gradW-(Pull(i,s)-W(i,s))/softness^2;
            %end
            for r=1:k
                gradW= gradW+HHt(s,r) * W(i,r); 
            end
            if(pullW==true)
                gradW=gradW-b*pullW_l1;
            elseif(pullH==true)
                gradW=gradW+b*pullW_l1;
            end
            if(proj_grad==true)
                if(W(i,s) == 0)
                    pg=min(0, gradW);
                else
                    pg=gradW;
                end
                violation = violation + pg^2;
            end
%             if(fpeak==true)
%                 hess = HHt(s,s)+gamma;%%1/softness^2;
%             else
            hess = HHt(s,s)+gamma;
%             end
            if(pullH==true || pullW==true)
                hess = hess+b;
            end
            %if(hess+gamma ~= 0)
                  W(i, s) = max(W(i, s) - gradW/(hess),0);
%                 W(i,s)=max((hess*W(i,s)-gradW)/(hess+gamma),0);
            %end
        end
    end
end
% if(a==0 & b==0 & step==1)
%     W_disp=W
%     H_disp=Ht'
% end
end


































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
%     AtW=A1'*W;
%     for s=1:k
%         %t = permutation[s]
%         for i=1:size(A,2)
%             gradHt = -AtW(i,s)+beta+delta*Ht(i,s);
%             for r=1:k
%                 gradHt = gradHt+WtW(s,r)*Ht(i,r);
%             end
% %             if(Ht(i,s) == 0)
% %                 pg=min(0, gradHt);
% %             else
% %                 pg=gradHt;
% %             end
% %             violation = violation + pg^2;
%             hess = WtW(s, s);
%             %if(hess+delta ~= 0)
%                 Ht(i, s) = max(Ht(i, s) - gradHt/(hess+delta), 0);
%             %end
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
%     for s=1:k
%         %t = permutation[s]
%         for i=1:size(W,1)
%             gradW= -AHt(i,s)+alpha+gamma*W(i,s);
%             for r=1:k
%                 gradW= gradW(i,s)+HHt(s,r) * W(i,r); 
%             end
%             if(W(i,s) == 0)
%                 pg=min(0, gradW);
%             else
%                 pg=gradW;
%             end
%             violation = violation + pg^2;
%             hess = HHt(s, s);
%             %if(hess+gamma ~= 0)
%                 W(i, s) = max(W(i, s) - gradW/(hess+gamma),0);
%             %end
%         end
%     end
% end