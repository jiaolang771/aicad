function L = convex_solution(B, aphla3)

m = size(B,2);

cvx_begin quiet
variable x(m,m) symmetric
minimise(vec(B)'*vec(x) + aphla3*vec(x)'*vec(x))
subject to
ones(1,m)*x == 0;
x - diag(diag(x)) <=0;
trace(x) == m;
cvx_end
L = x;
