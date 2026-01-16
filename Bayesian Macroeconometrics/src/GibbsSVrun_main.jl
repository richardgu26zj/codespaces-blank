function GibbsSVrun_main(y::Vector{Float64}, prior::GibbsSVprior;
                      Nsim::Int64=50_000, Nburn::Int64=5_000)
    @unpack ah0, bh0, atau, btau, nu0, S0 = prior
    T = length(y)

    # 1. Pre-allocate ALL buffers once (This kills the 17GiB leak)
    ystar   = Vector{Float64}(undef, T)
    # inv_s2j_buf = Vector{Float64}(undef, T)
    # rhs_buf     = Vector{Float64}(undef, T)
    
    # Pre-calculate Sparse Precision Matrix once
    H = sparse(I, T, T) - sparse(2:T, 1:T-1, ones(T-1), T, T)
    HH = H' * H

    store_h = Matrix{Float64}(undef, T, Nsim)
    store_para = Matrix{Float64}(undef, 3, Nsim)

    # Initial values
    h0 = log(var(y)) * 0.8
    h = fill(h0, T)
    tau = 0.0
    sigh2 = 0.1
    ah0_bh0 = ah0 / bh0
    atau_btau = atau / btau

    Random.seed!(124)

    @showprogress for isim in 1:(Nsim + Nburn)
        # STEP A: In-place ystar update (No new vector created)
        @inbounds for t in 1:T
            ystar[t] = log((y[t] - tau)^2 + 0.0001)
        end
        
        # STEP B: Sample h (Pass buffers to avoid internal allocations)
        h = SVRW_mod(ystar,h,h0,sigh2,HH)

        # STEP C: Sample h0 
        Kh0 = 1.0 / (1.0/sigh2 + 1.0/bh0)
        h0_hat = Kh0 * (h[1]/sigh2 + ah0_bh0)
        h0 = h0_hat + sqrt(Kh0) * randn()

        # STEP D: Sample tau (Reuse ystar_buf as a temporary container)
        sum_inv_sig = 0.0
        dot_inv_y = 0.0
        @inbounds for t in 1:T
            inv_sig = exp(-h[t])
            sum_inv_sig += inv_sig
            dot_inv_y += inv_sig * y[t]
        end
        Ktau = 1.0 / (sum_inv_sig + 1.0/btau)
        tau_hat = Ktau * (dot_inv_y + atau_btau)
        tau = tau_hat + sqrt(Ktau) * randn()

        # STEP E: Sample sigh2 (Manual SSR loop - avoids diff(h) allocation)
        SSR = (h[1] - h0)^2
        @inbounds for t in 2:T
            SSR += (h[t] - h[t-1])^2
        end
        sigh2 = rand(InverseGamma(nu0 + T/2, S0 + SSR/2))

        # STEP F: Storage (Broadcast with .= to modify the matrix slice)
        if isim > Nburn
            isave = isim - Nburn
            @views @inbounds for t in 1:T
                store_h[t, isave] = exp(h[t] / 2.0)
            end
            @views store_para[:, isave] .= [h0, tau, sigh2]
        end
    end 
    
    return store_h, store_para
end