function [loss,P,A,H] = ForwardPass(RNN,X,Y,h0,K)
    h=h0;
    n=size(X,2);
    P = zeros(K,n);
    m=size(RNN.W,1);
    H=zeros(m,n+1);
    A=zeros(m,n);
    loss = 0;
    H(:,1)=h;
    for t =1:n
        at=RNN.W*h+RNN.U*X(:,t) + RNN.b;
        A(:,t)=at;
        h=tanh(at);
        H(:,t+1)=h;
        o=RNN.V*h+RNN.c;
        p=exp(o)/sum(exp(o));
        P(:,t)=p;
        loss = loss-log(p(find(Y(:,t),1)));
    end
end