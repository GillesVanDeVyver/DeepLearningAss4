function grads = BackwardPass(RNN,P,A,H,X,Y,K)
    n=size(P,2);
    grad_o=zeros(K,n);
    m=size(RNN.W,1);
    grad_V=zeros(size(RNN.V));
    grad_W=zeros(size(RNN.W));
    grad_U=zeros(size(RNN.U));
    grad_b=zeros(size(RNN.b));
    grad_c=zeros(size(RNN.c));
    for t=n:-1:1
        grad_ot=-transpose((Y(:,t)-P(:,t)));
        grad_o(:,t)=grad_ot;
        grad_V=grad_V+transpose(grad_ot)*transpose(H(:,t+1));
        grad_c=grad_c+transpose(grad_ot);
        if t==n
            grad_ht=transpose(grad_o(:,n))*RNN.V;
        else
            grad_ht=transpose(grad_o(:,t))*RNN.V+grad_at*RNN.W;
        end
        grad_at=grad_ht*diag(1-(tanh(A(:,t))).^2);
        grad_W=grad_W+transpose(grad_at)*transpose(H(:,t));
        grad_U=grad_U+transpose(grad_at)*transpose(X(:,t));
        grad_b=grad_b+transpose(grad_at);
    end
    grads = struct('V',grad_V,'W',grad_W,'U',grad_U,'b',grad_b,'c',grad_c);
end