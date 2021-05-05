function synth_string= Synthesize(RNN,synth_length,h0,ind_to_char,K)
    x0=zeros(K,1); 
    x0(2)=1;
    h=h0;
    x=x0;
    Y=zeros(K,synth_length);
    synth_string="";
    for pos =1:synth_length
        a=RNN.W*h+RNN.U*x + RNN.b;
        h=tanh(a);
        o=RNN.V*h+RNN.c;
        p=exp(o)/sum(exp(o));
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        x = zeros(K,1);
        x(ii)=1;
        Y(:,pos)=x;
        ind = find(Y(:,pos),1);
        synth_string=synth_string+ind_to_char(ind);
    end
end