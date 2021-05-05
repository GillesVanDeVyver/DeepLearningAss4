rng(400);
book_fname = 'data/Goblet.txt';
[char_to_ind,ind_to_char,K,book_data] = ReadData(book_fname);

hyper_paras = struct('m',5,'eta',0.1,'seq_length',2);
RNN = InitParas(hyper_paras,K);


h0 = zeros(hyper_paras.m, 1);


X_chars = book_data(1:hyper_paras.seq_length);
Y_chars = book_data(2:hyper_paras.seq_length+1);
X = convertToOneHot(X_chars,K,char_to_ind);
Y = convertToOneHot(Y_chars,K,char_to_ind);






% gradient check
[loss,P,A,H] = ForwardPass(RNN,X,Y,h0,K);
grads = BackwardPass(RNN,P,A,H,X,Y,K);
step=1e-4;
num_grads = ComputeGradsNum(X, Y, RNN, step);
eps = 1e-3;
for f = fieldnames(num_grads)'
    assert(testSame(grads.(f{1}),num_grads.(f{1}), eps));
end


%Synthesize
x0=zeros(K,1);
x0(3)=1;
Y = Synthesize(RNN,x0,h0,hyper_paras.seq_length,K);
for pos =1:hyper_paras.seq_length
    ind = find(Y(:,pos),1);
    ind_to_char(ind);
end

