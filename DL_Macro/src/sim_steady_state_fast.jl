using LinearAlgebra

"""
Optimized equality check. 
"""
function equal_tolerance(x1, x_2, tol)::Bool
    length(x1) != length(x2) && return false
    @inbounds for i in eachindex(x1)
        if abs(x1[i] - x2[i]) >= tol
            return false
        end
    end
    return true
end

"""
compute the stationary distribution of Markov transition matrix Pi
"""
function stationary_markov(Pi :: Matrix{Float64}; tol = 1e-14)
    n = size(Pi, 1)
    pi_dist = fill(1.0/n, n)
    pi_new = similar(pi_dist)

    for it in 1:10_000
        # multiplies the transpose of Pi by the vector pi_dist
        mul!(pi_new, Pi', pi_dist) # \Pi_new = \Pi^\top\times pi_dist

        if it % 10 == 0 && equal_tolerance(pi_dist, pi_new, tol)
            return pi_new
        end
        copyto!(pi_dist, pi_new) # take the newly calculated distribution
        # (pi_new) and overwrite the old distribution (pi_dist)
        # so we can use it for the next round.
    end
    return pi_dist
end

"""
Monotonic interpolation
"""
function interpolate_monotonic!(y_out, x_target, xp, yp)
    nxp = length(xp)
    xp_i = 1

    @inbounds for xi in eachindex(x_target)
        xt = x_target[xi]
        # move the grid pointer forward
        while xp_i < nxp - 1 && xt >= xp[xp_i + 1]
            xp_i += 1
        end

        x_lo, x_hi = xp[xp_i], xp[xp_i +1]
        weight = (x_hi - xt) / (x_hi - x_lo)
        y_out[xi] = weight*yp[xp_i] + (1.0 - weight)*yp[xp_i + 1]
    end
end

"""
backward iteration step using the Endogenous Grid Method
"""
function backward_iteration!(Va_new, a_pol, c_pol, 
            Va, Pi, a_grid, y, r, beta, eis)
    # step 1: expected marginal utility
    # Pi*Va is a matrix-matrix multiplication
    Wa = beta .*(Pi*Va)

    # step 2: invert marginal utility to get endogenous consumption
    c_endog = Wa.^(-eis)

    # step 3: solve for policy using the grid of future assets
    # for each income state e, we interpolate back to the fixed asset grid
    for e in eachindex(y)
        # cash on hand today if we chose a_grid tomorrow
        coh_at_future_a = @views c_endog[e, :] .+ a_grid
        # the current actual cash on hand 
        actual_coh = y[e] .+(1+r) .*a_grid

        # we fine the assset policy by interpolating
        @views interpolate_monotonic!(a_pol[e, :], actual_coh, 
                        coh_at_future_a, a_grid)
        
        # enforce borrowing constraint 
        for j in eachindex(a_pol[e, :])
            if a_pol[e, j] < a_grid[1]
                a_pol[e, j] = a_grid[1]
            else
                break # monotonicity allows early break
            end
        end
    

        # back out consumption and update Va via Envelope condition
        @views c_pol[e, :]= (y[e] .+ (1+r) .*a_grid) .- a_pol[e, :]
        @views Va_new[e, :] .= (1+r) .*(c_pol[e, :].^(-1/eis))
    end
end

function distribution_ss(Pi, a_pol, a_grid; tol=1e-10)
    n_e, n_a = size(a_pol)
    D = fill(1.0 / (n_e * n_a), n_e, n_a)
    D_new = similar(D)
    
    # Pre-calculate lotteries to avoid re-computing indices in the loop
    # (Implementation of interpolate_lottery_loop omitted for brevity)
    
    for it in 1:10_000
        # Custom kernel to move mass based on policy a_pol
        forward_iteration_kernel!(D_new, D, Pi, a_pol, a_grid)
        
        if it % 10 == 0 && equal_tolerance(D, D_new, tol)
            return D_new
        end
        copyto!(D, D_new)
    end
    return D
end


