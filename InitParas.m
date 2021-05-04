function  RNN = InitParas(hyper_paras,K)
    b=zeros(hyper_paras.m,1);
    c=zeros(K,1);
    sig = .01;
    U = randn(hyper_paras.m, K)*sig;
    W = randn(hyper_paras.m, hyper_paras.m)*sig;
    V = randn(K, hyper_paras.m)*sig;
    h0=zeros(hyper_paras.m,1);
    RNN = struct('b',b,'c',c,'U',U,'W',W,'V',V,'h0',h0,'K',K);
end