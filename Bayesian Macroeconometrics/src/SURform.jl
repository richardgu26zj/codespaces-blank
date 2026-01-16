using SparseArrays, LinearAlgebra

n = 2; 
p = 2; 
T = 2; 
k = n*p + 1;
nk = n*k;

idi = repeat(1:n*T, inner = n*T)
idj = repeat(1:nk,2,1) 
X = [ones(n,1) randn(T,n*p)]
Xtmp = vec(repeat(X,1,2)')
Xout = sparse(idi,idj,Xtmp',n*T,nk)


row_indices = Int[]; col_indices = Int[]; values = Float64[];

for t in 1:T, i in 1:n
    row = (t-1)*n + i
    col_start = (i-1)*k
    for j in 1:k
        push!(row_indices, row)
        push!(col_indices, col_start + j)
        push!(values, X[t, j])
    end
end

Xout = sparse(row_indices, col_indices, values, n*T, nk)
display(Matrix(Xout))

idi = Int[]; idj = Int[]; xvalues = Float64[];

for t = 1:T, i = 1:n
    row = (t-1)*n + i 
    col = idi*k 
    for j = 1:k
        push!(idi, row)
        push!(idj, col+j)
        push!(xvalues, X[i,j])
    end
end

Xout = sparse(idi, idj,xvalues,n*T,T*n*k)