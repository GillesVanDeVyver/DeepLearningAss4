rng(400);
book_fname = 'data/Goblet.txt';
[char_to_ind,ind_to_char,K,book_data] = ReadData(book_fname);

hyper_paras = struct('m',100,'eta',0.1,'seq_length',2);
RNN = InitParas(hyper_paras,K);
x0=zeros(K,1);
x0(60)=1;



X_chars = book_data(1:hyper_paras.seq_length);
Y_chars = book_data(2:hyper_paras.seq_length+1);

X = convertToOneHot(X_chars,K,char_to_ind);
Y = convertToOneHot(Y_chars,K,char_to_ind);
[loss,P] = ForwardPass(RNN,X,Y);








n=25;
Y = Synthesize(RNN,x0,n);
for pos =1:n
    ind = find(Y(:,pos),1);
    ind_to_char(ind);
end

