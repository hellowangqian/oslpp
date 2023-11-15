function W = constructW(label)
W = zeros(length(label),length(label));
num_class = max(label(:));
for i = 1:num_class
    W = W + double(label==i)'*double(label==i);
end

W = W + double(label==-2)'*double(label==-2);

