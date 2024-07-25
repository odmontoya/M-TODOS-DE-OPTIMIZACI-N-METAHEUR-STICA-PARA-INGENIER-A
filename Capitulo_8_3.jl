# Capítulo 8.3 %%%% Algoritmo JAYA%%%%
using DataFrames, LinearAlgebra, Random;
using Statistics, Printf, BenchmarkTools;
B = [0.00013630 0.00006750 0.00007839
0.00006750 0.00015450 0.00009828
0.00007839 0.00009828 0.00016140];
Generadores = DataFrames.DataFrame([
(1, 0.006085, 10.04025, 136.9125, 5, 150),
(2, 0.005915, 9.760576, 59.15500, 15, 100),
(3, 0.005250, 8.662500, 328.1250, 50, 250),])
DataFrames.rename!(Generadores,
[:g, :a, :b, :c, :xmin, :xmax]);
a = Generadores.a; b = Generadores.b; c = Generadores.c;
xmin = Generadores.xmin;
xmax = Generadores.xmax;
Periodos = DataFrames.DataFrame([
(1, 210),(7, 150),(13,310),(19,450),
(2, 230),(8, 100),(14,320),(20,460),
(3, 200),(9, 80 ),(15,350),(21,470),
(4, 180),(10,130),(16,380),(22,420),
(5, 240),(11,190),(17,400),(23,350),
(6, 180),(12,280),(18,420),(24,220),])
DataFrames.rename!(Periodos, [:h, :PD]);
PD = Periodos.PD;
t = size(PD,1); g = size(Generadores,1);
H = 1:t; G = 1:g; alpha = 1;
function fitnessf(a,b,c,xmin,xmax,G,H,pg,PD,alpha,B)
    global Penalty = 100;
    global Penalization = 0;
    for h in H
        Pgen = sum(pg[k,h] for k in G)
        Pdl = PD[h] + alpha*sum(sum(pg[k,h]*B[k,m]*pg[m,h] for m in G) for k in G)
        if Pgen < Pdl
            global Penalization = Penalization - Penalty*min(Pgen - Pdl,0);
        end
    end
    for h in H
        for g in G
            if pg[g,h] < xmin[g]
                Penalization = Penalization + Penalty*max( xmin[g]-pg[g,h],0);
            end
            if pg[g,h] >xmax[g]
                Penalization = Penalization + Penalty*max(pg[g,h]-xmax[g],0);
            end
        end
    end
    Objval = sum(sum(a[k]*pg[k,h]^2+b[k]*pg[k,h]+c[k] for k in G) for h in H) + Penalization;
    return Objval
end
global tmax=5000; global Ns=120;
global Nv=72; global mmax=1000;
global m=0; global Nt = size(Periodos,1);
global lb = zeros(Nv,1); global ub = zeros(Nv,1)
for j in G
    global lb[(j-1)*Nt+1:j*Nt] = xmin[j]*ones(Nt,1);
    global ub[(j-1)*Nt+1:j*Nt] = xmax[j]*ones(Nt,1);
end
global fa=zeros(tmax,1);
function jaya(tmax,Ns,Nv,mmax,m,fa,xmin,xmax,lb,ub)
    for t=1:tmax
        if t==1
            global x=zeros(Ns,Nv+1);
            global cont=0;
            for e=1:Ns
                x[e,1:Nv]=lb+rand(Nv,1).*(ub-lb);
            end
            for e = 1:Ns
                for u = 1:Nv
                    if x[e,u] < lb[u,1]
                        global x[e,u] = lb[u,1];
                    elseif x[e,u] > ub[u,1]
                        global x[e,u] = ub[u,1];
                    end
                end
            end
            for i=1:Ns
                global aux=hcat(x[i,1:24],x[i,25:48],x[i,49:72]);
                 x[i,end]=fitnessf(a,b,c,xmin,xmax,G,H,aux',PD,alpha,B);
            end
            x=x[sortperm(x[:, Nv+1]), :];
            global peor_individuo=x[Ns,1:Nv];
            global mejor_individuo=x[1,1:Nv];
        end
        if t>=2
            global y=zeros(Ns,Nv+1);
            y[:,1:Nv]=x[:,1:Nv]+rand(Ns,Nv).*(repeat(mejor_individuo', outer = [Ns, 1, 1])-abs.(x[:,1:Nv]))-rand(Ns,Nv).*(repeat(peor_individuo', outer = [Ns, 1, 1])-abs.(x[:,1:Nv]));
            for e = 1:Ns
                for u = 1:Nv
                    if y[e,u] < lb[u,1]
                        global y[e,u] = lb[u,1];
                    elseif y[e,u] > ub[u,1]
                        global y[e,u] = ub[u,1];
                    end
                end
            end
            for i=1:Ns
                global aux=hcat(y[i,1:24],y[i,25:48],y[i,49:72]);
                y[i,end]=fitnessf(a,b,c,xmin,xmax,G,H,aux',PD,alpha,B);
            end
            for i=1:Ns
                if y[i,end]<x[i,end]
                    x[i,:]=y[i,:];
                end
            end
            x=x[sortperm(x[:, Nv+1]), :];
            global peor_individuo=x[Ns,1:Nv];
            global mejor_individuo=x[1,1:Nv];
        end
        global fa[t,1]=x[1,Nv+1];
        if t>=2
            if fa[t-1,1]==fa[t,1]
                m=m+1;
            else
                m=0;
            end
        end
        if m==mmax
            break;
        end
    end
    return x[1,:];
end
t1 = time();
rep = 1;
global Guardar = zeros(rep,Nv+1);
for r = 1:rep
    global Guardar[r,:] = jaya(tmax,Ns,Nv,mmax,m,fa,xmin,xmax,lb,ub);
end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
elapsed_time = time() - t1;
@printf("Valor maximo: %2.1f
\n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.1f
\n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.1f
\n",mean(Guardar[:,Nv+1]))
@printf("Desviacion estandar: %2.1f
\n",std(Guardar[:,Nv+1]))
@printf("Maximo - Media: %2.1f \n",
Guardar[rep,Nv+1]-mean(Guardar[:,Nv+1]));
println("Tiempo de ejecucion (promedio): ",
round(elapsed_time/rep, digits = 4), " s");
           
            

    
    

