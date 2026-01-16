using SparseArrays, Distributions, LinearAlgebra, Random

function SVRW_mod(ystar::AbstractVector{Float64}, h::AbstractVector{Float64},
                  h0::Real,sigh2::Real,HH::AbstractArray{Float64})
    T = length(h)
    pj = [0.00730 0.10556 0.00002 0.04395 0.34001 0.24566 0.25750]; 
	mj = [-10.12999 -3.97281 -8.56686 2.77786 0.61942 1.79518 -1.08819] .- 1.2704; # mean adjusted
	s2j = [5.79596 2.61369 5.17950 0.16735 0.64009 0.34023 1.26261]; 
	sj = sqrt.(s2j); 

    P = pj .*pdf.(Normal.(h .+ mj, sj), ystar)
    Prob = P./sum(P, dims = 2)
    vec(S) = 7 .- sum(rand(T) .< cumsum(Prob, dims=2)) .+1
    
    iSig = Diagonal(vec(1 ./s2j[S]))
    Dh = iSig + HH/sigh2
    h_hat = Dh\(iSig*(ystar .-mj[S])+h0/sigh2*HH*ones)
    h = h_hat + cholesky(Symmetric(Dh)).L'\randn(T)

    return vec(h)
    
end