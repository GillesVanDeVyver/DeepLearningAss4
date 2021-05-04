function [loss,P] = ForwardPass(RNN,X,Y)
    h=RNN.h0;
    n=size(X,2);
    P = zeros(RNN.K,n);
    loss = 0;
    for pos =1:n
        a=RNN.W*h+RNN.U*X(:,pos) + RNN.b;
        h=tanh(a);
        o=RNN.V*h+RNN.c;
        p=exp(o)/sum(exp(o));
        P(:,pos)=p;
        loss = loss-log(p(find(Y(:,pos),1)));
    end
end