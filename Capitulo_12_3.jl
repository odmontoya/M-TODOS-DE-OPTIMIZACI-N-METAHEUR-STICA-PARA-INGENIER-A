# Capítulo 12.3 %%%%% Algoritmo por enjambre de Salpas %%%
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
Ns = 1500;
Nv = 2*n;
tmax = 10000;
global ymin=0;
global ymax=1
function SSA(tmax,Ns,Nv,n,ymax,ymin,F,xmax,xmin,X)
    for t=2:tmax
        global c1 = 2*exp(-(4*t/tmax)^2);
        for i=1:Ns
            if i<Ns/2
                for j=1:Nv
                    global c2=rand();
                    global c3=rand();
                    if j<=n && c3<=0.5
                        X[i,j]=F[1,j]+c1*((ymax-ymin)*c2+ymin)
                    elseif j<=n && c3>0.5
                        X[i,j]=F[1,j]-c1*((ymax-ymin)*c2+ymin)
                    elseif j>n && c3<=0.5
                        X[i,j]=F[1,j]+c1*((xmax[j-n]-xmin[j-n])*c2+xmin[j-n])
                        if X[i,j]>xmax[j-n]
                            X[i,j]=xmax[j-n]
                        elseif X[i,j]<xmin[j-n]
                            X[i,j]=xmin[j-n]
                        end
                    elseif j>n && c3>=0.5
                        X[i,j]=F[1,j]-c1*((xmax[j-n]-xmin[j-n])*c2+xmin[j-n])
                        if X[i,j]>xmax[j-n]
                            X[i,j]=xmax[j-n]
                        elseif X[i,j]<xmin[j-n]
                            X[i,j]=xmin[j-n]
                        end
                    end
                    if j<=n && X[i,j]<0.5
                        X[i,j]=0
                    elseif j<=n && X[i,j]>=0.5
                        X[i,j]=1
                    end
                end
            else
                X[i,1:Nv]= (X[i,1:Nv].+ X[i-1,1:Nv])./2
                for j=1:n
                    if X[i,j]<0.5
                        X[i,j]=0
                    else
                        X[i,j]=1
                    end
                end
            end
            global y = X[i,1:n];
            global x = X[i,n+1:Nv];
            global X[i,Int(Nv+1)] = fitnessf(a,b,c,n,y,x,N);
        end
        X = X[sortperm(X[:, Nv+1]), :];
        if X[1,Nv+1]< F[1,Nv+1]
            F[1,:]=X[1,:];
        end
    end
    return F
end
t1 = time();
rep = 100;
global Guardar = zeros(rep,Nv+1);
for r = 1:rep
    global X = rand(Float64,Ns,Nv+1);
    for j = 1:Ns
        for k = 1:Nv
            if k <= n
                X[j,k] = rand(0:1);
            else
                X[j,k] = xmin[k-n] + rand(Float64)*(xmax[k-n]-xmin[k-n]);
            end
        end
        global y = X[j,1:n];
        global x = X[j,n+1:Nv];
        global X[j,Int(Nv+1)] = fitnessf(a,b,c,n,y,x,N);
    end
    X = X[sortperm(X[:, Nv+1]), :];
    global F=zeros(1,Nv+1);
    F[1,:]=X[1,:];
    global Guardar[r,:]=
    SSA(tmax,Ns,Nv,n,ymax,ymin,F,xmax,xmin,X);
    println([r, Guardar[r,Nv+1]])
end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
et = time() - t1;
@printf("Valor maximo: %2.4f\n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.4f\n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.4f\n",mean(Guardar[:,Nv+1]))
@printf("Desviacion estandar: %2.4f\n",std(Guardar[:,Nv+1]))
@printf("|Minimo - Media|: %2.4f\n",abs(Guardar[1,Nv+1]-mean(Guardar[:,Nv+1])))
println("Tiempo promedio: ",round(et/rep, digits = 4), " s");
y = Guardar[1,1:n]; x = Guardar[1,n+1:Nv];
@printf("Binaria\tContinua\n")
for k in N
    @printf("%0.0f\t%0.5f \n",y[k],x[k])
end