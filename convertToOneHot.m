function X = convertToOneHot(X_chars,K,char_to_ind)
    n=size(X_chars,2);
    X=zeros(K,n);
    for pos =1:n
        X(char_to_ind(X_chars(pos)),pos)=1;
    end

end