function [W,H]= initializewh(A,k)
% Copied from sklearn.decomposition.nmf._initialize_nmf
% Based off Boutsidis and Gallopoulos 2008
% A: data matrix, k: target rank
% W,H are initial factor matrix guesses.
epsilon=1*10^(-7);
p=10;
[U,S,V]=LOCAL_rsvd(A,k,p);
[m,n]=size(A);
W=zeros(m,k);
H=zeros(k,n);
W(:,1)=sqrt(S(1,1))*abs(U(:,1)); %First column of W
H(1,:)=sqrt(S(1,1))*abs(V(:,1)'); %First row of H
x_p=zeros(m,1);
x_n=zeros(m,1);
y_p=zeros(1,n);
y_n=zeros(1,n);
for p=2:k
    x=U(:,p);
    y=V(:,p)';
    for i=1:m
        x_p(i)=max(0,x(i));
        x_n(i)=abs(min(0,x(i)));
    end
    for j=1:n
        y_p(j)=max(0,y(j));
        y_n(j)=abs(min(0,y(j)));  
    end
    pos_norm=norm(x_p)*norm(y_p);
    neg_norm=norm(x_n)*norm(y_n);
    if (pos_norm>neg_norm)
            u = x_p / norm(x_p);
            v = y_p / norm(y_p);
            sigma = pos_norm;
    else
            u = x_n / norm(x_n);
            v = y_n / norm(y_n);
            sigma = neg_norm;
    end
    W(:,p)=sqrt(S(p,p)*sigma)*u;
    H(p,:)=sqrt(S(p,p)*sigma)*v;
end
default=1/10*mean(mean(A));
%default=mean(mean(A));
default=0;
W(W<epsilon)=default;
H(H<epsilon)=default;
end
            
            
            
            
            
            