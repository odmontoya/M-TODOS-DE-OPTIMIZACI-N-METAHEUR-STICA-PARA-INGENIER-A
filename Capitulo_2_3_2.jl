#Capítulo 2.3.2
using JuMP, Ipopt, Juniper, DataFrames
Parametros = DataFrames.DataFrame([
(1, 5, 10, 0.11, -6, 5),(2, 1, 9, 0.50, -6, 8),
(3, 9, 10, 0.61, -13, 10),(4, 1, 4, 0.17, -7, 12),
(5, 7, 1, 0.44, -15, 5),(6, 2, 7, 0.56, -9, 9),
(7, 2, 1, 0.45, -11, 11),(8, 8, 4, 0.60, -5, 5),
(9, 3, 5, 0.79, -14, 14),(10, 1, 9, 0.51,-8, 14),]);
DataFrames.rename!(Parametros,[:k,:a,:b,:c,:xmin,:xmax]);
a = Parametros.a; b = Parametros.b; c = Parametros.c;
xmin = Parametros.xmin; xmax = Parametros.xmax;
n = size(Parametros,1); N = 1:n;
optimizer = Juniper.Optimizer
nl_solver = optimizer_with_attributes(Ipopt.Optimizer)
MINLP = Model(optimizer_with_attributes(optimizer,"nl_solver"=>nl_solver))
@variable(MINLP, xmin[k]<=x[k in N]<= xmax[k])
@variable(MINLP, y[k in N],Bin)
@NLobjective(MINLP,Min,sum(y[k]*(a[k]*sin(b[k]*x[k]) + c[k]*x[k]^2) for k in N)) # me tocó poner NLconstraint y NLobjective
@NLconstraint(MINLP,sum(y[k] for k in N) <= n/2)
JuMP.optimize!(MINLP)
using Printf
@printf("Binaria\tContinua\n")
for k in N
@printf("%0.0f\t%0.5f \n",
abs.(value.(y[k])),value.(x[k]))
end