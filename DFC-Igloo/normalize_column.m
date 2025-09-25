function A = normalize_column(A)
[~,ncolumn] = size(A);
for i = 1 : ncolumn
    temp1 = mean(A(:,i));
    temp2 = std(A(:,i));
    A(:,i) = (A(:,i) - temp1)/temp2;
end
