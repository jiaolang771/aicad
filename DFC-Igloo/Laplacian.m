function L = Laplacian(A)

L = diag(sum(A))-A;

end
