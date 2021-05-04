book_fname = 'data/Goblet.txt';
[char_to_ind,ind_to_char,K] = ReadData(book_fname);

hyper_paras = struct('m',100,'eta',0.1,'seq_length',25);
RNN = InitParas(hyper_paras,K);

