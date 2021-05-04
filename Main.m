rng(400);
book_fname = 'data/Goblet.txt';
[char_to_ind,ind_to_char,K] = ReadData(book_fname);

hyper_paras = struct('m',100,'eta',0.1,'seq_length',25);
RNN = InitParas(hyper_paras,K);
h0=randn(hyper_paras.m,1);
x0=zeros(K,1);
x0(60)=1;
n=3;


Y = Synthesize(RNN,h0,x0,n,K);
for pos =1:n
    ind = find(Y(:,pos),1);
    ind_to_char(ind)
end
