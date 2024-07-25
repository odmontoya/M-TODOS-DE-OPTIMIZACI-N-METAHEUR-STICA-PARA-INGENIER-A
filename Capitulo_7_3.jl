# Capítulo 7.3
using DataFrames, LinearAlgebra, Random;
using Printf, Statistics, BenchmarkTools;
B = [0.00013630 0.00006750 0.00007839
0.00006750 0.00015450 0.00009828
0.00007839 0.00009828 0.00016140];
Generadores = DataFrames.DataFrame([
(1, 0.006085, 10.04025, 136.9125, 5, 150),
(2, 0.005915, 9.760576, 59.15500, 15, 100),
(3, 0.005250, 8.662500, 328.1250, 50, 250),]);
DataFrames.rename!(Generadores,[:g,:a,:b,:c,:pmin,:pmax]);
a = Generadores.a; b = Generadores.b; c = Generadores.c;
pmin = Generadores.pmin; pmax = Generadores.pmax;
Periodos = DataFrames.DataFrame([
(1, 210),(2, 230),(3, 200),(4, 180),
(5, 240),(6, 180),(7, 150),(8, 100),
(9, 80 ),(10,130),(11,190),(12,280),
(13,310),(14,320),(15,350),(16,380),
(17,400),(18,420),(19,450),(20,460),
(21,470),(22,420),(23,350),(24,220),]);
DataFrames.rename!(Periodos, [:h, :PD]);
PD = Periodos.PD;
h=size(Periodos.h,1);
g = size(Generadores,1);
H = 1:h;
G = 1:g;
alpha=1;
Nv=g*h;
Ns=1000;
rep=100;
global tmax=5000;
global fl=3.5;
global AP= 0.05;
function fitnessf(a,b,c,G,H,pg,PD,alpha,B)
    Penalty = 35;
    Penalization = 0;
    for h in H
        Pgen = sum(pg[k,h] for k in G)
        Pdl = PD[h] +alpha*sum(sum(pg[k,h]*B[k,m]*pg[m,h] for m in G) for k in G)
        if Pgen < Pdl
            Penalization = Penalization + abs(Penalty*(Pgen - Pdl))+Penalty;
        end
    end
    Objval = sum(sum(a[k]*pg[k,h]^2+b[k]*pg[k,h]+c[k] for k in G) for h in H) + Penalization;
    return Objval
end
function Pob_Mem_inicial(g,h,Nv,Ns,G,pmin,pmax)
    xo=zeros(Ns,Nv+1)
    mo=zeros(Ns,Nv+1)
    for i=1:Ns
        global Pg_x=zeros(g,h)
        global Pg_m=zeros(g,h)
        for k in G
            Pg_x[k,1:h]=rand(pmin[k]:pmax[k],1,h)
            Pg_m[k,1:h]=rand(pmin[k]:pmax[k],1,h)
            if k==1
                xo[i,1:k*h]=Pg_x[k,1:h]
                mo[i,1:k*h]=Pg_m[k,1:h]
            elseif k==2
                xo[i,h+1:k*h]=Pg_x[k,1:h]
                mo[i,h+1:k*h]=Pg_m[k,1:h]
            else
                xo[i,2*h+1:k*h]=Pg_x[k,1:h]
                mo[i,2*h+1:k*h]=Pg_m[k,1:h]
            end
        end
        xo[i,Nv+1]=fitnessf(a,b,c,G,H,Pg_x,PD,alpha,B)
        mo[i,Nv+1]=fitnessf(a,b,c,G,H,Pg_m,PD,alpha,B)
    end
    return mo, xo
end
function CSA(mt,xt,Nv,Ns,g,h,AP,fl,pmin,pmax,G)
    for i=1:Ns
        global pos=rand(1:Ns,1)
        global xi=zeros(Nv)
        global Pg_x=zeros(g,h)
        if rand() > AP
            xi = xt[i,1:Nv] + (fl*rand()).*(mt[pos[1],1:Nv] - xt[i,1:Nv])
            for k=1:g
                if k==1
                    Pg_x[k,1:h]= xi[1:k*h]
                elseif k==2
                    Pg_x[k,1:h]= xi[h+1:k*h]
                else
                    Pg_x[k,1:h]= xi[2*h+1:k*h]
                end
            end
            cont=0
            for k=1:h
                for m=1:g
                    if Pg_x[m,k]< pmin[m] || Pg_x[m,k]>pmax[m]
                        cont=cont+1
                    end
                end
            end
            if cont==0
                xt[i,1:Nv]=xi
                xt[i,Nv+1]=fitnessf(a,b,c,G,H,Pg_x,PD,alpha,B)
                if xt[i,Nv+1]<mt[i,Nv+1]
                    mt[i,1:Nv]=xt[i,1:Nv]
                    mt[i,Nv+1]=xt[i,Nv+1]
                end
            end
        else
            for k in G
                Pg_x[k,1:h]=rand(pmin[k]:pmax[k],1,h)
                if k==1
                    xt[i,1:k*h]=Pg_x[k,1:h]
                elseif k==2
                    xt[i,h+1:k*h]=Pg_x[k,1:h]
                else
                    xt[i,2*h+1:k*h]=Pg_x[k,1:h]
                end
            end
            xt[i,Nv+1]=fitnessf(a,b,c,G,H,Pg_x,PD,alpha,B)
            if xt[i,Nv+1]<mt[i,Nv+1]
                mt[i,1:Nv]=xt[i,1:Nv]
                mt[i,Nv+1]=xt[i,Nv+1]
            end
        end
    end
    return mt, xt
end
t1 = time();
global Guardar = zeros(rep,Nv+1);
for r=1:rep
    global m, x =Pob_Mem_inicial(g,h,Nv,Ns,G,pmin,pmax);
    for j=1:tmax
        m, x = CSA(m,x,Nv,Ns,g,h,AP,fl,pmin,pmax,G);
    end
    m=m[sortperm(m[:, Nv+1]), :];
    global Guardar[r,:] = m[1,:];
end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
et = time() - t1;
@printf("Valor maximo: %2.1f\n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.1f\n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.1f\n",mean(Guardar[:,Nv+1]))
@printf("Desviacion estandar: %2.1f\n",std(Guardar[:,Nv+1]))
@printf("Media - Minimo: %2.1f \n",mean(Guardar[:,Nv+1])-Guardar[1,Nv+1])
println("Tiempo promedio: ",round(et/rep, digits = 4), " s");
println("xbest:");
Pg_m=zeros(g,h)
for k=1:g
    if k==1
        local Pg_m[k,1:h]= Guardar[1,1:k*h]
    elseif k==2
        Pg_m[k,1:h]= Guardar[1,h+1:k*h]
    else
        Pg_m[k,1:h]= Guardar[1,2*h+1:k*h]
    end
end
for k=1:g
    @printf("Gen. %d \t",k)
end
@printf("\n")
for i in H
    for k in G
        @printf("%0.5f \t",Pg_m[k,i])
    end
    @printf("\n")
end