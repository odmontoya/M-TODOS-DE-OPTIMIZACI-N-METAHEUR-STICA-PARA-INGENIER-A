# Capítulo 4.3 %%% Algoritmo de senos y cosenos %%%
using DataFrames
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
Vmax = 861; Wmax = 1215;n = size(C,2); P = 1:n;
function fitnessf(C,V,W,Vmax,Wmax,x,P)
    Penalty = 100;
    Vol = sum(V[p]*x[p] for p in P);
    Wei = sum(W[p]*x[p] for p in P);
    Objval = sum(C[p]*x[p] for p in P) -
    Penalty*(max(Vol - Vmax,0) + max(Wei - Wmax,0));
    return Objval
end
function ASC(X,lb,ub,tmax,a,Nv,Ns)
    for t in 1:tmax
        global xbest = X[Ns,:];
        for k in 1:Ns
            global r1 = a*(1 - t/tmax)
            global r2 = 2*pi*rand(Float64,(Nv+1,1));
            global r3 = rand(Float64);
            global r4 = rand(Float64);
            if isless(r4,1/2)
                global y = X[k,:]+
                r1*sin.(r2).*abs.(r3.*xbest-X[k,:]);
            else
                global y = X[k,:]+
                r1*cos.(r2).*abs.(r3.*xbest-X[k,:]);
            end
             global y = (round.(y));
            for j = 1:Nv
                if y[j,1] < lb[j,1]
                    global y[j,1] = lb[j,1];
                elseif y[j,1] > ub[j,1]
                    global y[j,1] = ub[j,1];
                end
            end
            global y[Nv+1,1] =
            fitnessf(C,V,W,Vmax,Wmax,y[1:Nv,1],P);
            if X[k,Nv+1] < y[Nv+1,1]
                global X[k,:] = y;
            end
        end
        global X = X[sortperm(X[:, Nv+1]), :];
    end
    return xbest
end
using Printf, Statistics, BenchmarkTools
t1 = time(); rep = 100;
Ns = 1000; Nv = n; tmax = 100; a = 2;
lb = zeros(Nv,1); ub = ones(Nv,1);
global Guardar = zeros(rep,Nv+1);
    for r = 1:rep
        global X = rand(Float64,Ns,Nv+1);
        X = round.(X);
        for k in 1:Ns
            local x = X[k,1:Nv];
            global X[k,Nv+1] =
            fitnessf(C,V,W,Vmax,Wmax,x,P);
        end
        global X = X[sortperm(X[:, Nv+1]), :];
        ASC(X,lb,ub,tmax,a,Nv,Ns)
        global Guardar[r,:] = xbest;
    end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
et = time() - t1;
@printf("Valor maximo: %2.1f
\n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.1f
\n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.1f
\n",mean(Guardar[:,Nv+1]))
@printf("Desviacion estandar: %2.1f
\n",std(Guardar[:,Nv+1]))
@printf("Maximo - Media: %2.1f
\n",Guardar[rep,Nv+1]-mean(Guardar[:,Nv+1]))
println("Tiempo promedio: ",
round(et/rep, digits = 4), " s");
println("xbest:", Int.(Guardar[rep,1:Nv])');