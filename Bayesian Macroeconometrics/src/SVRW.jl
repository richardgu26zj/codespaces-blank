using SparseArrays, Distributions, LinearAlgebra, Random

function SVRW(ystar::Vector{Float64}, h::Vector{Float64}, h0::Real, sigh2::Real)

    T = length(h)
	pj = [0.00730 0.10556 0.00002 0.04395 0.34001 0.24566 0.25750]; 
	mj = [-10.12999 -3.97281 -8.56686 2.77786 0.61942 1.79518 -1.08819] .- 1.2704; # mean adjusted
	s2j = [5.79596 2.61369 5.17950 0.16735 0.64009 0.34023 1.26261]; 
	sj = sqrt.(s2j); 

    P = pj .*pdf.(Normal.(h .+mj, sj), ystar)
    P = P ./sum(P, dims=2)
    S = 7 .- sum(rand(T) .< cumsum(P, dims=2), dims=2) .+1

    H = sparse(I,T,T) - sparse(2:T, 1:T-1, vec(ones(1,T-1)), T, T)
    HH = H'*H
    #iSig = sparse(1:T, 1:T, vec(1 ./s2j[S]))
    #iSig = spdiagm(0 => vec(1 ./s2j[S]))
    iSig = Diagonal(vec(1 ./s2j[S]))
    Dh = iSig + HH/sigh2
    h_hat = Dh\(iSig*(ystar .-mj[S]) + h0/sigh2*HH*ones(T,1))
    #h = h_hat + cholesky(Dh).L'\randn(T)
    h = h_hat + cholesky(Dh).L'\randn(T,1) 

    return h 
end
