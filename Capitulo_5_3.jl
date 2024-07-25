# Capítulo 5.3 %%%% Algoritmo de aprendizaje incremental basado en poblaciones %%
using JuMP, HiGHS, DataFrames, LinearAlgebra
using Random, Statistics, Printf, BenchmarkTools
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
Vmax = 861; Wmax = 1215; n = size(C,2); P = 1:n;
function fitnessf(C,V,W,Vmax,Wmax,x)
    Penalty = 100
    global Vol = sum(V.*x);
    global Wei = sum(W.*x)
    Objval = sum(C.*x)- Penalty*(max(Vol .- Vmax,0)
    + max(Wei .- Wmax,0));
    return Objval
end
Ns=80; n=2; Nv=30; LRmax=1; LRmin=0; Etol=0.001;
function PBIL(Ns,n,Nv,LRmax,LRmin,Etol)
    Pob=zeros(Ns,Nv);
    global PM=ones(n,Nv)*(1/n);
    global NI=round.(PM*Ns);
    for i=1:Nv
        local aux=Ns-NI[1,i];
        global NI[2,i]=aux;
    end
    global x=zeros(Ns,Nv);
    for i=1:Ns
        for j=1:Nv
            local aux=rand((0,1));
            local aux2=0;
            if ((aux==0)&&(NI[1,j]!=0)&&(aux2==0))
                x[i,j]=0;
                NI[1,j]=NI[1,j]-1;
                aux2=1;
            end
            if ((aux==1)&&(NI[2,j]!=0)&&(aux2==0))
                x[i,j]=1;
                NI[2,j]=NI[2,j]-1;
                aux2=1;
            end
            if ((aux==0)&&(NI[1,j]==0)&&(aux2==0))
                x[i,j]=1;
                NI[2,j]=NI[2,j]-1;
                aux2=1;
            end
            if ((aux==1)&&(NI[2,j]==0)&&(aux2==0))
                x[i,j]=0;
                NI[1,j]=NI[1,j]-1;
                aux2=1;
            end
        end
    end
    global fo=zeros(Ns,1);
    for i=1:Ns
        fo[i,1]=fitnessf(C',V',W',Vmax,Wmax,x[i,:]);
    end
    global mejorsol, mejorpos = findmax(fo);
    global mejorpos=x[mejorpos[1],:];
    global E=0;
    for i=1:n
        for j=1:Nv
            global E=E+PM[i,j]*log2(PM[i,j]);
        end
    end
    E=-E/Nv;
    global t=1;
    while (E>=Etol)
        global t=t+1;
        global LR=LRmax-((LRmax-LRmin)/(1+exp(-10*(E-0.5))));
        for i=1:n
            for j=1:Nv
                aux=mejorpos[j,1];
                if(aux==0)
                    if (i==1)
                        local PMact=PM[i,j]+(1-PM[i,j])*LR;
                        PM[i,j]=PMact;
                    end
                    if(i==2)
                        local PMact=PM[i,j]+(1-PM[i,j])*LR;
                        PM[i,j]=(1-PMact)*((PM[i,j])/(1-PM[i,j]));
                    end
                end
                if(aux==1)
                    if(i==1)
                        local PMact=PM[i,j]+(1-PM[i,j])*LR;
                        PM[i,j]=(1-PMact)*((PM[i,j])/(1-PM[i,j]));
                    end
                    if (i==2)
                        local PMact=PM[i,j]+(1-PM[i,j])*LR;
                        PM[i,j]=PMact;
                    end
                end
            end
        end
        global NI=round.(PM*Ns);
        for i=1:Nv
            local aux=Ns-NI[1,i];
            global NI[2,i]=aux;
        end
        for i=1:Ns
            for j=1:Nv
                local aux=rand((0,1));
                local aux2=0;
                if ((aux==0)&&(NI[1,j]!=0)&&(aux2==0))
                    x[i,j]=0;
                    NI[1,j]=NI[1,j]-1;
                    aux2=1;
                end
                if ((aux==1)&&(NI[2,j]!=0)&&(aux2==0))
                    x[i,j]=1;
                    NI[2,j]=NI[2,j]-1;
                    aux2=1;
                end
                if ((aux==0)&&(NI[1,j]==0)&&(aux2==0))
                    x[i,j]=1;
                    NI[2,j]=NI[2,j]-1;
                    aux2=1;
                end
                if ((aux==1)&&(NI[2,j]==0)&&(aux2==0))
                    x[i,j]=0;
                    NI[1,j]=NI[1,j]-1;
                    aux2=1;
                end
            end
        end
        global fo=zeros(Ns,1);
        for i=1:Ns
            fo[i,1]=fitnessf(C',V',W',Vmax,Wmax,x[i,:]);
        end
        global mejorsol, mejorpos = findmax(fo);
        global mejorpos=x[mejorpos[1],:];
        global E=0;
        for i=1:n
            for j=1:Nv
                global E=E+PM[i,j]*log2(PM[i,j]);
            end
        end
        E=-E/Nv;
    end
    global Xbest=zeros(1,Nv+1);
    Xbest[1,Nv+1]=mejorsol;
    for i=1:Nv
        if (PM[1,i]>=PM[2,i])
            Xbest[1,i]=0;
        else
            Xbest[1,i]=1;
        end
    end
    return Xbest;
end
t1 = time();
rep = 100;
global Guardar = zeros(rep,Nv+1);
    for r = 1:rep
        PBIL(Ns,n,Nv,LRmax,LRmin,Etol)
        global Guardar[r,:] = Xbest;
    end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
elapsed_time = time() - t1;
@printf("Valor maximo: %2.1f \n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.1f \n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.1f\n",mean(Guardar[:,Nv+1]))
@printf("Desviacion estandar: %2.1f\n",std(Guardar[:,Nv+1]))
@printf("Maximo - Media: %2.1f\n",Guardar[rep,Nv+1]-mean(Guardar[:,Nv+1]))
println("Tiempo de ejecucion (promedio): ",
round(elapsed_time/rep, digits = 4), " s");
println("xbest: ", Int.(Guardar[rep,1:Nv])');