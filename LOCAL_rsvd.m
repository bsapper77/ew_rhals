%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the "basic" randomized sampling algorithm.
% It works well when the svds of A decay rapidly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,D,V] = LOCAL_rsvd(A,k,p)
%A: data matrix
%k: Number of factors
%p: oversampling

n         = size(A,2); %Number of columns
ell       = k + p; %Number of samples to take from column space
Omega     = randn(n,ell); % k+p random normal vectors of length n
Y         = A*Omega; % Y contains random samples of the column space of A
[Q,~,~]   = qr(Y,0); %qr(Y,0) is an economy qr decomposition (if rows>cols, only first n cols of Q are find, and first n rows of R are found)
%If rows<=cols, qr and economy qr are same
B         = Q'*A; %Project data into a lower dimension
[UU,D,V]  = svd(B,'econ'); %Take svd of B
U         = Q*UU(:,1:k); %Project UU back into higher dimension
D         = D(1:k,1:k); %Singular values
V         = V(:,1:k); %Right Singular Vectors

return