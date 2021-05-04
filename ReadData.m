function  [char_to_ind,ind_to_char] = ReadData(book_fname)
    fid = fopen(book_fname,'r');
    book_data = fscanf(fid,'%c');
    fclose(fid);

    book_chars=unique(book_data);
    K = numel(book_chars);

    %char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    %ind_to_char = containers.Map('KeyType','int32','ValueType','char');

    values = zeros(1,K);
    keys = cell(1,K);
    for i = 1:length(book_chars)
      keys(i)={book_chars(i)};
      values(i)=i;
    end
    char_to_ind = containers.Map(keys,values);
    ind_to_char = containers.Map(values,keys);
end