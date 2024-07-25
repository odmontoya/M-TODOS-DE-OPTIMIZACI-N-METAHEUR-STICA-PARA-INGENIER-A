#Capítulo 2.1.1
using JuMP, HiGHS, DataFrames
ks = DataFrames.DataFrame([
(236, 40, 67),(484, 45, 34),(500, 50, 83),(136, 25, 49),
(380, 28, 33),(299, 25, 75),(481, 50, 17),(419, 35, 34),
(419, 21, 92),(301, 30, 15),(226, 31, 62),(467, 41, 19),
(224, 29, 64),(202, 39, 45),(474, 21, 91),(241, 30, 34),
(287, 39, 85),(209, 21, 68),(319, 43, 23),(231, 50, 45),
(111, 47, 86),(182, 37, 60),(233, 45, 84),(339, 39, 60),
(412, 31, 92),(251, 45, 43),(460, 29, 19),(235, 21, 51),
(341, 43, 78),(439, 47, 13),])
DataFrames.rename!(ks, [:C, :V, :W])
C = ks.C'; V = ks.V'; W = ks.W';
Vmax = 861; Wmax = 1215;
n = size(C,2); N = 1:n;
mochila = Model(HiGHS.Optimizer)
@variable(mochila,x[k in N],Bin)
@objective(mochila,Max,sum(C[k]*x[k] for k in N))
@constraint(mochila,sum(W[k]*x[k] for k in N) <= Wmax)
@constraint(mochila,sum(V[k]*x[k] for k in N) <= Vmax)
JuMP.optimize!(mochila)
println(Int.(round.(value.(x)))')