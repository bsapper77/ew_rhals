function [W,H]= initializefactors(A,k,initialize)
%A(matrix): the data matrix to be initialized
%k(int): the number of factors
%initialize(string): Either random for a random initialization, nndsvd for an svd initialization, or nndsvdmean for an svd initialization with the mean of the data 
if (initialize=="random")
    W=abs(randn(size(A,1),k)); %initialize with absolute value of random normals, no justification for a specific distribution
    H=abs(randn(size(A,2),k))'; %initialize
    mA=mean((A),'all'); %mean of A
    W=mA*W; %scale factor matrices by the mean of A
    H=mA*H; %Scale factor matrices by the mean of A
%     W=abs(rand(size(A,1),k)); %Absolute value of random uniform variables
%     H=abs(rand(size(A,2),k))';
end
if (initialize=="nndsvd")
    mn=false; %don't use mean for default values
    [W,H]=svdinit(A,k,mn); %Svd initialization
end
if (initialize=="nndsvdmean")
    mn=true; %Use mean for default values
    [W,H]=svdinit(A,k,mn);
end
end