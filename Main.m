rng(400);
book_fname = 'data/Goblet.txt';
[char_to_ind,ind_to_char,K,book_data] = ReadData(book_fname);

hyper_paras = struct('m',100,'eta',0.1,'seq_length',25,'eps',1e-2,'n_epochs',2);
plotTitle = strcat('m=',string(hyper_paras.m),',eta=',string(hyper_paras.eta),...
            ',seq length=',string(hyper_paras.seq_length),',eps=',string(hyper_paras.eps),...
            ',n epochs=',string(hyper_paras.n_epochs)) 
RNN = InitParas(hyper_paras,K);
h0 = zeros(hyper_paras.m, 1);

book_data_sample=book_data(1,1:5000);


[RNN,smooth_loss] = trainAdaGrad(RNN,hyper_paras,book_data_sample,char_to_ind,ind_to_char,h0,K,plotTitle);





%{
fileID = fopen(strcat(plotTitle,'_final_synth.txt'),'w');
synth_string= Synthesize(RNN,1000,h0,ind_to_char,K);
fprintf(fileID,synth_string);
fclose(fileID);
%}



%{
hyper_paras = struct('m',100,'eta',0.1,'seq_length',25,'eps',1e-2,'n_epochs',2);
RNN = InitParas(hyper_paras,K);
% sample for checks
X_chars = book_data(1:hyper_paras.seq_length);
Y_chars = book_data(2:hyper_paras.seq_length+1);
X = convertToOneHot(X_chars,K,char_to_ind);
Y = convertToOneHot(Y_chars,K,char_to_ind);

% gradient check
[loss,P,A,H] = ForwardPass(RNN,X,Y,h0,K);
grads = BackwardPass(RNN,P,A,H,X,Y,K);
step=1e-4;
num_grads = ComputeGradsNum(X, Y, RNN, step);
eps = 1e-5;
for f = fieldnames(num_grads)'
    assert(testSame(grads.(f{1}),num_grads.(f{1}), eps));
end
%}
