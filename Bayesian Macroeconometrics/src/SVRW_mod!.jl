using StaticArrays, LinearAlgebra, Random, SparseArrays

# Pre-calculate log constants once outside to save time
const LOG_PJ = SVector{7, Float64}(log(0.00730), log(0.10556), log(0.00002), log(0.04395), log(0.34001), log(0.24566), log(0.25750))
const LOG_S2J = SVector{7, Float64}(log(5.79596), log(2.61369), log(5.17950), log(0.16735), log(0.64009), log(0.34023), log(1.26261))

function SVRW_mod!(h::Vector{Float64}, ystar::Vector{Float64}, h0::Float64, sigh2::Float64, 
                   HH::SparseMatrixCSC{Float64, Int64}, inv_s2j_S::Vector{Float64}, rhs::Vector{Float64})
    
    T = length(h)
    mj  = SVector{7, Float64}(-11.39999, -5.24321, -9.83726, 1.50746, -0.65098, 0.52478, -2.35859)
    s2j = SVector{7, Float64}(5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261)

    # STEP 1: Indicator Sampling
    @inbounds for t in 1:T
        max_lp = -Inf
        lps = MVector{7, Float64}(undef)
        
        for j in 1:7
            diff = ystar[t] - (h[t] + mj[j])
            lp = LOG_PJ[j] - 0.5*LOG_S2J[j] - 0.5*(diff^2 / s2j[j])
            lps[j] = lp
            if lp > max_lp; max_lp = lp; end
        end
        
        sum_p = 0.0
        for j in 1:7
            lps[j] = exp(lps[j] - max_lp)
            sum_p += lps[j]
        end
        
        u = rand() * sum_p
        curr_p = 0.0
        chosen_s = 7
        for j in 1:7
            curr_p += lps[j]
            if u <= curr_p
                chosen_s = j; break
            end
        end
        
        inv_s2j_S[t] = 1.0 / s2j[chosen_s]
        rhs[t] = inv_s2j_S[t] * (ystar[t] - mj[chosen_s])
    end

    # STEP 2: Boundary Condition
    # In a Random Walk, h[1] = h0 + err. The precision contribution is 1/sigh2
    rhs[1] += h0 / sigh2

    # STEP 3: Matrix Solve
    # We add a tiny epsilon (1e-12) to ensure positivity
    # The 'HH / sigh2' term must be the sparse H'H matrix
    Dh = Symmetric(Diagonal(inv_s2j_S) + HH / sigh2 + I * 1e-12)

    try
        C = cholesky(Dh)
        # Solve for the mean
        h_hat = C \ rhs
        
        # Draw the innovation correctly using the Sparse factor
        # innovation = P' * inv(L') * z
        z = randn(T)
        innovation = C.L' \ z 
        
        # Update h in place
        h .= h_hat .+ innovation
    catch
        @warn "Cholesky failed: check if sigh2 is negative or HH is singular"
    end
    
    return h
end