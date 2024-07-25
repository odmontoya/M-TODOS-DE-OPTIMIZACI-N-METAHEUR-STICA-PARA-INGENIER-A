# Capítulo 11.3 %%%% Algoritmo PSO %%%
using DataFrames
Parametros = DataFrames.DataFrame([
(1,5,10,0.11, -6, 5),(2, 1,9,0.50,-6, 8),
(3,9,10,0.61,-13,10),(4, 1,4,0.17,-7,12),
(5,7, 1,0.44,-15, 5),(6, 2,7,0.56,-9, 9),
(7,2, 1,0.45,-11,11),(8, 8,4,0.60,-5, 5),
(9,3, 5,0.79,-14,14),(10,1,9,0.51,-8,14),]);
DataFrames.rename!(Parametros,[:k, :a, :b, :c, :xmin, :xmax]);
a = Parametros.a; b = Parametros.b; c = Parametros.c;
xmin = Parametros.xmin; xmax = Parametros.xmax;
n = size(Parametros,1); N = 1:n;
function fitnessf(a,b,c,n,y,x,N)
    Penalty = 100;
    Objval=(sum(y[k]*(a[k]*sin(b[k]*x[k])+c[k]*x[k]^2) for k in N) + Penalty*(max(sum(y[k] for k in N)-n/2,0)));
    return Objval
end
global tmax=1000; global Ns=1000; global Nv=2*n;
global X=zeros(Ns,Nv+1); global Inermax=1;
global Inermin=0; global Phi1=1.494;
global Phi2=1.494; global Velmax=0.1;
global Velmin=-Velmax; contnm=0; nmmax=100;
function pso(Ns,Nv,tmax,Inermax,Inermin,Phi1,Phi2,Velmax,Velmin)
    global X=zeros(Ns,Nv+1);
    for t=1:tmax
        if t==1
            global X = rand(Float64,Ns,Nv+1);
            for j = 1:Ns
                for k = 1:Nv
                    if k <= n
                        X[j,k] = rand(0:1);
                    else
                        X[j,k] = xmin[k-n] + rand(Float64)*(xmax[k-n]-xmin[k-n]);
                    end
                end
            end
            global Vel= Velmin.*ones(Ns,Nv)+(Velmax-Velmin).*rand(Ns,Nv);
            X[:,1:Nv]=X[:,1:Nv]+Vel;
            X[:,1:n] = abs.(round.(X[:,1:n]));
            for j=1:Ns
                for m = 1:Nv
                    if m <= n
                        if X[j,m] > 1
                            global X[j,m] = 1;
                        elseif X[j,m]< 0
                            global X[j,m] = 0;
                        end
                    else
                        if X[j,m]< xmin[m-n]
                            global X[j,m] = xmin[m-n];
                        elseif X[j,m] > xmax[m-n]
                            global X[j,m] = xmax[m-n];
                        end
                    end
                end
            end
            for j=1:Ns
                global y1 = X[j,1:n]
                global x1 = X[j,n+1:Nv];
                global X[j,Int(Nv+1)] = fitnessf(a,b,c,n,y1,x1,N)
            end
            global mejorposi=X[:,1:Nv];
            global aptitudi=X[:,Nv+1];
            global aptitudg,
            mejorposg = findmin(aptitudi);
            global mejorposg=X[mejorposg[1],1:Nv];
        end
        if t>=2
            global Inercia=Inermax-(((Inermax-Inermin)*t)/tmax)
            Vel=Inercia*Vel+(Phi1.*rand(Ns,Nv)).*(mejorposi-X[:,1:Nv])+(Phi2*rand(Ns,Nv)).*(repeat(mejorposg', outer = [Ns, 1, 1])-X[:,1:Nv]);
            X[:,1:Nv]=X[:,1:Nv]+Vel;
            X[:,1:n] = abs.(round.(X[:,1:n]));
            for j=1:Ns
                for m = 1:Nv
                    if m <= n
                        if X[j,m] > 1
                            global X[j,m] = 1;
                        elseif X[j,m]< 0
                            global X[j,m] = 0;
                        end
                    else
                        if X[j,m]< xmin[m-n]
                            global X[j,m] = xmin[m-n];
                        elseif X[j,m] > xmax[m-n]
                            global X[j,m] = xmax[m-n];
                        end
                    end
                end
            end
            global aux=1;
            for j=1:Ns
                global y1 = X[j,1:n]
                global x1 = X[j,n+1:Nv];
                global X[j,Int(Nv+1)] = fitnessf(a,b,c,n,y1,x1,N)
                if X[j,Int(Nv+1)]<aptitudi[j,1] 
                    aptitudi[j,1]=X[j,Int(Nv+1)];
                    mejorposi[j,:]=X[j,1:Nv];
                end
                if X[j,Int(Nv+1)]<aptitudg
                    aptitudg=X[j,Int(Nv+1)];
                    mejorposg=X[j,1:Nv];
                    global aux=0;
                end
                if aux==1
                    global contnm=contnm+aux;
                else
                    global contnm=0;
                end
            end
        end
        if contnm==nmmax
            break;
        end
    end
    global xbest=[mejorposg;aptitudg];
    return xbest;
end
t1 = time(); rep = 100;
global Guardar = zeros(rep,Nv+1);
for r = 1:rep
    pso(Ns,Nv,tmax,Inermax,Inermin,Phi1,Phi2,Velmax,Velmin)
    global Guardar[r,:] = xbest;
    println([r, Guardar[r,Nv+1]])
end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
et = time() - t1;
@printf("Valor maximo: %2.4f\n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.4f\n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.4f\n",mean(Guardar[:,Nv+1]))
@printf("Desv. estandar: %2.4f\n",std(Guardar[:,Nv+1]))
@printf("|Minimo - Media|: %2.4f\n",abs(Guardar[1,Nv+1]-mean(Guardar[:,Nv+1])))
println("Tiempo promedio: ",round(et/rep, digits = 4), " s");
y = Guardar[1,1:n]; x = Guardar[1,n+1:Nv];
@printf("Binaria\tContinua\n")
for k in N
    @printf("%0.0f\t%0.5f \n",y[k],x[k])
end