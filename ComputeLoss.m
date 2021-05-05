function loss = ComputeLoss(X, Y, RNN, h0)
    K=size(RNN.U,2);
    [loss,~,~,~] = ForwardPass(RNN,X,Y,h0,K);
end