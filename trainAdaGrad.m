function [RNN,smooth_loss] = trainAdaGrad(RNN,hyper_paras,book_data,char_to_ind,ind_to_char,h0,K,plotTitle)
    plot_interval=100;
    moments = struct('b',zeros(size(RNN.b)),'c',zeros(size(RNN.c)),'U',zeros(size(RNN.U)),'W',zeros(size(RNN.W)),'V',zeros(size(RNN.V)));    
    nb_iterations=floor(length(book_data)/hyper_paras.seq_length);
    nb_eval_points=(ceil(nb_iterations/plot_interval))*hyper_paras.n_epochs;
    loss_vector=zeros(1,nb_eval_points);
    plot_iter=1;
    %for synth    
    synth_interval=2;
    synth_length=200;
    fileID = fopen(strcat(plotTitle,'_synth.txt'),'w');
    for epoch=1:hyper_paras.n_epochs
        epoch
        e=1;
        iter=1;
        hprev=h0;
        while e<length(book_data)-hyper_paras.seq_length-1
            X_chars_batch = book_data(e:e+hyper_paras.seq_length-1);
            Y_chars_bacth = book_data(e+1:e+hyper_paras.seq_length);
            X_batch = convertToOneHot(X_chars_batch,K,char_to_ind);
            Y_batch = convertToOneHot(Y_chars_bacth,K,char_to_ind);
            [loss,P,A,H] = ForwardPass(RNN,X_batch,Y_batch,hprev,K);
            grads = BackwardPass(RNN,P,A,H,X_batch,Y_batch,K);
            for f = fieldnames(RNN)'
                moments.(f{1})=moments.(f{1})+grads.(f{1}).^2;
                RNN.(f{1})= RNN.(f{1})-(hyper_paras.eta./sqrt(moments.(f{1})+hyper_paras.eps)).*grads.(f{1});
                grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
            end
            hprev = H(:,hyper_paras.seq_length+1);
            if plot_iter == 1
                smooth_loss=loss;
            else
                smooth_loss = .999*smooth_loss + .001 * loss;
            end
            if mod(iter,plot_interval)==1
                loss_vector(plot_iter)=smooth_loss;
                if mod(plot_iter,synth_interval)==1
                    synth_string= Synthesize(RNN,synth_length,h0,ind_to_char,K);
                    result = strcat('iter=',string(iter),', smooth_loss=',string(smooth_loss),':\n',synth_string,'\n\n');
                    fprintf(fileID,result);
                end
                plot_iter=plot_iter+1;                
            end
            e=e+hyper_paras.seq_length;
            iter=iter+1;
        end
    end
    plot_inds = 1:plot_interval:nb_eval_points*plot_interval;
    figure
    plot(plot_inds,loss_vector)
    xlabel('iteration') 
    ylabel('loss')
    legend({'training loss'},'Location','northeast')
    title(plotTitle)
    ylim([0 max(loss_vector)+20])
    xlim([1 max(plot_inds)])
    saveas(1,strcat(strrep(plotTitle, '.', ','),'.png'))
    fclose(fileID);
end