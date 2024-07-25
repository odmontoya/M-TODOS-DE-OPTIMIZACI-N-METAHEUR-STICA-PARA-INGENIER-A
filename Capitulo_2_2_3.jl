#Capítulo 2.2.3
using JuMP, Ipopt, DataFrames
B = [0.00013630 0.00006750 0.00007839
0.00006750 0.00015450 0.00009828
0.00007839 0.00009828 0.00016140];
Generadores = DataFrames.DataFrame([
(1, 0.006085, 10.04025, 136.9125, 5, 150),
(2, 0.005915, 9.760576, 59.15500, 15, 100),
(3, 0.005250, 8.662500, 328.1250, 50, 250),])
DataFrames.rename!(Generadores, [:g,:a,:b,:c,:pmin,:pmax]);
a = Generadores.a; b = Generadores.b; c = Generadores.c;
pmin = Generadores.pmin; pmax = Generadores.pmax;
Periodos = DataFrames.DataFrame([
(1, 210),(7, 150),(13,310),(19,450),
(2, 230),(8, 100),(14,320),(20,460),
(3, 200),(9, 80 ),(15,350),(21,470),
(4, 180),(10,130),(16,380),(22,420),
(5, 240),(11,190),(17,400),(23,350),
(6, 180),(12,280),(18,420),(24,220),])
DataFrames.rename!(Periodos, [:h, :PD]);
PD = Periodos.PD; t = size(PD,1);
g = size(Generadores,1); H = 1:t; G = 1:g;
alpha = 1;
DE = Model(Ipopt.Optimizer)
@variable(DE,pmin[k] <= Pg[k in G, h in H] <= pmax[k])
@NLobjective(DE,Min,sum(sum(a[k]*Pg[k,h]^2+
b[k]*Pg[k,h]+c[k] for k in G) for h in H))
for h in H
@NLconstraint(DE,sum(Pg[k,h] for k in G) == PD[h] +
alpha*sum(sum(Pg[k,h]*B[k,m]*Pg[m,h] for m in G)
for k in G));
end
JuMP.optimize!(DE)
@show objective_value(DE)
using Printf
for k in G
@printf("Gen. %d \t",k)
end
@printf("\n")
for h in H
for k in G
@printf("%0.5f \t",value.(Pg[k,h]))
end
@printf("\n")
end