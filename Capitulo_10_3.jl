# Capítulo 10.3 %%%% Algoritmo de optimización de distribución normal generalizada %%%
using DataFrames
Parametros = DataFrames.DataFrame([
(1,5,10,0.11, -6, 5),(2, 1, 9, 0.50,-6, 8),
(3,9,10,0.61,-13,10),(4, 1, 4, 0.17,-7,12),
(5,7, 1,0.44,-15, 5),(6, 2, 7, 0.56,-9, 9),
(7,2, 1,0.45,-11,11),(8, 8, 4, 0.60,-5, 5),
(9,3, 5,0.79,-14,14),(10,1, 9, 0.51,-8,14),]);
DataFrames.rename!(Parametros,[:k,:a,:b,:c,:xmin,:xmax]);
a = Parametros.a; b = Parametros.b; c = Parametros.c;
xmin = Parametros.xmin; xmax = Parametros.xmax;
n = size(Parametros,1); N = 1:n;
function fitnessf(a,b,c,n,y,x,N)
    Penalty = 100;
    Objval=(sum(y[k]*(a[k]*sin(b[k]*x[k])+c[k]*x[k]^2) for k in N) + Penalty*(max(sum(y[k] for k in N)-n/2,0)));
    return Objval
end
function GNDO(X,Nv,Ns,tmax,xmax,xmin)
    for t = 1:tmax
        global xbest = X[1,:]
        for k = 1:Ns
            global delta = rand(Float64)
            global xkt = X[k,:];
            r1 = rand(Float64); r2 = rand(Float64);
            global lambda1 = rand(Float64,1,Nv+1)';
            global lambda2 = rand(Float64,1,Nv+1)';
            if r1 <= r2
                global eta = sqrt.(-log.(lambda1)).*cos.(2*pi.*lambda2);
            else
                global eta = sqrt.(-log.(lambda1)).*cos.(2*pi*lambda2 + pi.*ones(Nv+1,1));
            end
            beta = rand(Float64);
            global lambda3 = rand(Float64,1,Nv+1)';
            global lambda4 = rand(Float64,1,Nv+1)';
            if delta <= 1/2
                global yt = (1/Ns)*sum(X; dims=1)';
                global ukt = (1/3)*(xbest + xkt + yt);
                global sigmakt = sqrt.((xkt - ukt).^2 + (xbest - ukt).^2 + (yt - ukt).^2);
                global vkt = ukt + eta.*sigmakt;
            else
                p1 = rand(1:Ns); p2 = rand(1:Ns);
                p3 = rand(1:Ns);
                while (p1 == k || p2 == k || p3 == k ||
                    p1 == p2 || p1 == p3 || p2 == p3)
                    p1 = rand(1:Ns); p2 = rand(1:Ns);
                    p3 = rand(1:Ns);
                end
                xp1t = X[p1,:]; xp2t = X[p2,:];
                xp3t = X[p3,:];
                if xkt[Nv+1] <= xp1t[Nv+1]
                    global v1 = xkt - xp1t;
                else
                    global v1 = xp1t - xkt;
                end
                if xp2t[Nv+1] <= xp3t[Nv+1]
                    global v2 = xp2t - xp3t;
                else
                    global v2 = xp3t - xp2t;
                end
                global vkt = beta*abs.(lambda3).*v1 + (1-beta)*abs.(lambda4).*v2;
            end
            global vkt[1:n] = abs.(round.(vkt[1:n]));
            for m = 1:Nv
                if m <= n
                    if vkt[m] > 1
                        global vkt[m] = 1;
                    elseif vkt[m] < 0
                        global vkt[m] = 0;
                    end
                else
                    if vkt[m] < xmin[m-n]
                        global vkt[m] = xmin[m-n];
                    elseif vkt[m] > xmax[m-n]
                        global vkt[m] = xmax[m-n];
                    end
                end
            end
            global y1 = vkt[1:n];
            global x1 = vkt[n+1:Nv];
            global Objval = fitnessf(a,b,c,n,y1,x1,N);
            global vkt[Nv+1] = Objval;
            if Objval < X[k,Nv+1];
                X[k,:] = vkt
            end
        end
        X = X[sortperm(X[:, Nv+1]), :]
    end
    return xbest
end
using Printf, Statistics, BenchmarkTools, Random
t1 = time(); rep = 100;
Ns = 10000; Nv = 2*n; tmax = 100;
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
        global y = X[j,1:n]
        global x = X[j,n+1:Nv];
        global X[j,Int(Nv+1)] = fitnessf(a,b,c,n,y,x,N)
    end
    X = X[sortperm(X[:, Nv+1]), :]
    GNDO(X,Nv,Ns,tmax,xmax,xmin)
    global Guardar[r,:] = xbest;
    println([r, Guardar[r,Nv+1]])
end
global Guardar = Guardar[sortperm(Guardar[:, Nv+1]), :];
et = time() - t1;
@printf("Valor maximo: %2.4f\n",Guardar[rep,Nv+1])
@printf("Valor minimo: %2.4f\n",Guardar[1,Nv+1])
@printf("Valor promedio: %2.4f\n",mean(Guardar[:,Nv+1]))
@printf("Desv. estandar: %2.4f\n",std(Guardar[:,Nv+1]))
@printf("|Minimo - Media|: %2.4f\n",abs(Guardar[1,Nv+1]- mean(Guardar[:,Nv+1])))
println("Tiempo promedio: ",round(et/rep, digits = 4), " s");
y = Guardar[1,1:n]; x = Guardar[1,n+1:Nv];
@printf("Binaria\tContinua\n")
for k in N
    @printf("%0.0f\t%0.5f \n",y[k],x[k])
end
