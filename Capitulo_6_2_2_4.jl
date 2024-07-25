# Capítulo 6.2.2.4
using Printf; LinearAlgebra; Random;
using DataFrames; Statistics;
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
function fitnessf(C,V,W,Vmax,Wmax,x,P)
    Penalty = 100;
    Vol = sum(V[p]*x[p] for p in P);
    Wei = sum(W[p]*x[p] for p in P);
    Objval = sum(C[p]*x[p] for p in P) - Penalty*(max(Vol - Vmax,0) + max(Wei - Wmax,0));
    return Objval
end
function seleccion(Nv,Ns,x,np)
    ganadores=[]
    padres1= rand(1:Ns,np)
    for i in 1:2:length(padres1)
        if x[padres1[i],Nv+1] >= x[padres1[i+1],Nv+1]
            push!(ganadores, padres1[i])
        else
            push!(ganadores, padres1[i+1])
        end
    end
    return ganadores
end
function recombinacion(Nv,p,padres,C,W,Wmax,V,Vmax)
    padre1=p[padres[1],:]
    padre2=p[padres[2],:]
    corte=rand(2:Nv-1)
    hijo1=zeros(Float64,1,Nv+1)
    hijo1[1:corte]=padre1[1:corte]
    hijo1[corte+1:Nv]=padre2[corte+1:Nv]
    hijo1[Nv+1]=fitnessf(C,V,W,Vmax,Wmax,hijo1[1:Nv],P)
    return hijo1
end
function mutacion(hijo,Nv,C,W,Wmax,V,Vmax)
    genmutado=rand(1:Nv)
    if hijo[genmutado]==0
        hijo[genmutado]=1
    else
        hijo[genmutado]=0
    end
    hijo[Nv+1]=fitnessf(C,V,W,Vmax,Wmax,hijo[1:n],P)
    return hijo
end
Nv = size(C,2); P = 1:Nv; Ns=50;
rep = 100; tmax=1000; div=3; np=4;
global Guardar = zeros(rep,Nv+1);
for r=1:rep
    global x=round.(rand(Float64, (Ns, Nv+1)));
    for i=1:Ns
        x[i,Nv+1]=fitnessf(C,V,W,Vmax,Wmax,x[i,1:Nv],P);
    end
    x=x[sortperm(x[:, Nv+1]), :];
    for i=1:tmax
        y=zeros(1,Nv+1);
        global padres=seleccion(Nv,Ns,x,np);
        Hijor=recombinacion(Nv,x,padres,C,W,Wmax,V,Vmax);
        global y=mutacion(Hijor,Nv,C,W,Wmax,V,Vmax);
            if x[1,Nv+1]<y[Nv+1]
                global cont=0;
                for j=1:Ns
                    if x[j,1:Nv]==y[1:Nv];
                        cont=cont+1;
                    end
                end
                if cont<div
                    global x[1,1:end]=y[1:end];
                    global x=x[sortperm(x[:, Nv+1]), :];
                end
            end
    end
    global Guardar[r,:] = x[Ns,1:Nv+1];
end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
elapsed_time = time() - t1;
@printf("Valor maximo: %2.1f\n",Guardar[rep,Nv+1]);
@printf("Valor minimo: %2.1f\n",Guardar[1,Nv+1]);
@printf("Valor promedio: %2.1f\n",mean(Guardar[:,Nv+1]));
@printf("Desviacion estandar: %2.1f\n",std(Guardar[:,Nv+1]));
@printf("Maximo - Media: %2.1f\n",Guardar[rep,Nv+1]-mean(Guardar[:,Nv+1]));
println("Tiempo de ejecucion (promedio): ",
round(elapsed_time/rep, digits = 4), " s");
println("xbest: ", Int.(Guardar[rep,1:Nv])');



