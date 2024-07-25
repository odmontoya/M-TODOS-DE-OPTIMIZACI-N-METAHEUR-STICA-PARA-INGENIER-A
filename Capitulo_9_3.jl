#Capítulo 9.3 %%%% Optmización de Búsqueda por Vórtices%%%%
using DataFrames, Random, Distributions
B = [0.00013630 0.00006750 0.00007839
    0.00006750 0.00015450 0.00009828
    0.00007839 0.00009828 0.00016140];
Generadores = DataFrames.DataFrame([
    (1, 0.006085, 10.04025, 136.9125, 5, 150),
    (2, 0.005915, 9.760576, 59.15500, 15, 100),
    (3, 0.005250, 8.662500, 328.1250, 50, 250),])
DataFrames.rename!(Generadores,
[:g, :a, :b, :c, :pmin, :pmax]);
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
PD = Periodos.PD;
t = size(PD,1); g = size(Generadores,1);
H = 1:t; G = 1:g; alpha = 1;
function fitnessf(a,b,c,pmin,pmax,G,H,pg)
    Penalty = 1000; Penalization = 0;
    for h in H
        Pgen = round((sum(pg[k,h] for k in G)), digits = 4)
        Pdl = round((PD[h] +    alpha*sum(sum(pg[k,h]*B[k,m]*pg[m,h] for m in G) for k in G)), digits = 4)
        if Pgen < Pdl
            Penalization = Penalization + Penalty*abs(Pdl - Pgen) + Penalty;
        end
        for k in G
            if round(pg[k,h], digits = 4) < pmin[k,1]
                Penalization = Penalization + Penalty*abs(pmin[k,1]-pg[k,h]) + Penalty;
            end
            if round(pg[k,h], digits = 4) > pmax[k,1]
                Penalization = Penalization +Penalty*abs(pg[k,h] - pmax[k,1]) + Penalty;
            end
        end
    end
    Objval = sum(sum(a[k]*pg[k,h]^2+b[k]*pg[k,h]+c[k] for k in G) for h in H) + Penalization;
    return Objval
end

function ABV(Ns,Nv,tmax,lb,ub)
    global Mu = (1/2)*(lb + ub);
    global gmin = Inf; beta = 1e-3;
    for t = 1:tmax
        global r = (1+beta-t/tmax)*exp(-6*t/tmax)*(ub-lb).*rand(Nv,1)/2;
        global Sigma = zeros(Nv,Nv);
        for k = 1:Nv
            global Sigma[k,k] = r[k];
        end
        global X = rand(MvNormal(Mu[:,1], Sigma),Ns)';
        global Objvalx = zeros(Ns,1);
        for k = 1:Ns
            for j = 1:Nv
                if X[k,j] < lb[j,1]
                    global X[k,j] = lb[j,1];
                elseif X[k,j] > ub[j,1]
                    global X[k,j] = ub[j,1];
                end
            end
            global xaux = zeros(Ng,Nt);
            for j in G
                global xaux[j,:] = X[k,(j-1)*Nt+1:j*Nt]
            end
            global Objvalx[k,1] =
            fitnessf(a,b,c,pmin,pmax,G,H,xaux);
        end
        global minpos = argmin(Objvalx);
        global fmin = Objvalx[argmin(Objvalx)];
        global Iterbest = X[minpos[1],:];
        if fmin < gmin
            global gmin = fmin;
            global xbest = Iterbest;
            global Mu = xbest;
        end
        #println([t, gmin])
    end
    return xbest, gmin
end
using Printf, Statistics, BenchmarkTools;
t1 = time(); rep = 100;
Ns = 1000; Ng = size(Generadores,1);
Nt = size(Periodos,1); Nv = Ng*Nt;
global tmax = 5000;
lb = zeros(Nv,1); ub = zeros(Nv,1)
for j in G
    global lb[(j-1)*Nt+1:j*Nt] = pmin[j]*ones(Nt,1);
    global ub[(j-1)*Nt+1:j*Nt] = pmax[j]*ones(Nt,1);
end
global Guardar = zeros(rep,Nv+1);
for r = 1:rep
    ABV(Ns,Nv,tmax,lb,ub)
    global Guardar[r,1:Nv] = xbest;
    global Guardar[r,Nv+1] = gmin;
end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
et = time() - t1;
@printf("Valor maximo: %2.1f\n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.1f\n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.1f\n",mean(Guardar[:,Nv+1]))
@printf("Desviacion estandar: %2.1f\n",std(Guardar[:,Nv+1]))
@printf("Maximo - Media: %2.1f\n",Guardar[rep,Nv+1]-mean(Guardar[:,Nv+1]))
println("Tiempo promedio: ",
round(et/rep, digits = 4), " s");
Power = Guardar[1,1:Nv]';
Pg = zeros(Ng,Nt);
for k in G
    @printf("Gen. %d \t",k)
    Pg[k,:] = Power[(k-1)*Nt+1:k*Nt];
end
@printf("\n")
for h in H
    for k in G
        @printf("%0.5f \t",Pg[k,h])
    end
    @printf("\n")
end



    
    
