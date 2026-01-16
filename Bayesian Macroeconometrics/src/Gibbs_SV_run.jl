using SparseArrays,Random,Parameters,LinearAlgebra,Distributions

#includet("SVRW_mod.jl")

function Gibbs_SV_run(y::Vector{Float64},prior::GibbsSVprior;
                   Nsim::Int64=100_000, Nburn::Int64=5_000)
    # unpack prior setup 
    @unpack ah0, bh0, atau, btau, nu0, S0 = prior
    T = length(y)

    # calculate a few things before the loop 
    ah0_bh0 = ah0/bh0
    atau_btau = atau/btau
    H = sparse(I,T,T) - sparse(2:T,1:T-1,vec(ones(1,T-1)),T,T)
    HH = H'*H

    # ystar = Vector{Float64}(undef,T)
    # inv_sig_vec = Vector{Float64}(undef,T)

    store_h = Matrix{Float64}(undef, T, Nsim)
    store_para = Matrix{Float64}(undef, 3, Nsim)

    h0 = log(var(y))*0.8
    h = fill(h0,T)
    tau = 3
    sigh2 = 0.1

    Random.seed!(124)


    @showprogress for isim in 1:(Nsim + Nburn)
        @views begin
            # sample h 
            ystar = log.((y .- tau).^2 .+ 0.0001)
            h = SVRW_mod(ystar,h,h0,sigh2,HH)

            # sample h0 
            Kh0 = 1/(1/sigh2 + 1/bh0)
            h0_hat = Kh0*(h[1]/sigh2 + ah0_bh0)
            h0 = h0_hat + sqrt(Kh0)*randn()

            # sample tau 
            #iSig = sparse(1:T,1:T,vec(exp.(-h)))
            # iSig = Diagonal(exp.(-vec(h)))
            # Ktau = 1.0/(dot(ones(T),iSig,ones(T)) + 1.0/btau)
            # tau_hat = Ktau*(dot(ones(T),iSig,y) + atau_btau)
            inv_sig_vec = exp.(-h)
            Ktau = 1.0/(sum(inv_sig_vec) + 1.0/btau)
            tau_hat = Ktau*(dot(inv_sig_vec,y) + atau_btau)
            tau = tau_hat + sqrt(Ktau)*randn()

            # sample sigh2
            #SSR = dot(h-[h0;h[1:end-1]],h-[h0;h[1:end-1]])
            # SSR = (h[1]-h0)^2
            # for t in 2:T
            #     SSR +=(h[t]-h[t-1])^2
            # end
            SSR = (h[1]-h0)^2 + sum(abs2,diff(h))
            sigh2 = rand(InverseGamma(nu0+T/2, S0+SSR/2))

            if isim > Nburn
                isave = isim - Nburn
                store_h[:,isave] = exp.(h./2)
                store_para[:,isave] = [h0, tau, sigh2]
            end # end of storage
        end
    end # end of MCMC loop
    
    return store_h, store_para

end

