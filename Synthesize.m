function Y = Synthesize(RNN,x0,n)
    h=RNN.h0;
    x=x0;
    Y=zeros(RNN.K,n);
    for pos =1:n
        a=RNN.W*h+RNN.U*x + RNN.b;
        h=tanh(a);
        o=RNN.V*h+RNN.c;
        p=exp(o)/sum(exp(o));
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        x = zeros(RNN.K,1);
        x(ii)=1;
        Y(:,pos)=x;
    end
end