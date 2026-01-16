using Random, LinearAlgebra

"""
     This function is designed for Durbin and Koopman smoother
    with stochastic volatility modification 

    Simulation smoother for h_t conditional on mixture indicators.
    
"""

function dk_smoother_sv(y_adj::Vector{Float64}, varj_vec::Vector{Float64},
                                        phi::Float64, sigh2::Float64)
    
   T = length(y_adj)

   # ---- Step 1: Draw simulated data (h^+, y^+)
   h_plus = Vector{Float64}(undef, T)
   y_plus = Vector{Float64}(undef, T)

   # draw initial state from unconditional distribution
   h_curr = randn()*sqrt(sigh2/(1-phi^2))
   @inbounds for t in 1:T 
            h_plus[t] = h_curr
            y_plus[t] = h_curr + randn()*sqrt(varj_vec[t])
            if t < T 
                h_curr = phi*h_curr + randn()*sqrt(sigh2)
            end
        end

  # --- step 2: run Kalman filter on diff
  y_diff = y_adj .- y_plus

  v = Vector{Float64}(undef, T)   # prediction error
  F = Vector{Float64}(undef, T)   # prediction error variance 
  K = Vector{Float64}(undef, T)   # Kalman gain

  s_pred = 0.0
  p_pred = sigh2/(1-phi^2)

  @inbounds for t in 1:T 
          v[t] = y_diff[t] - s_pred
          F[t] = p_pred + varj_vec[t]
          K[t] = p_pred/F[t]

          s_pred = phi*(s_pred + K[t]*v[t])
          p_pred = phi^2*p_pred*(1-K[t]) + sigh2
  end

# --- step 3: backward smoothing pass
   r = 0.0
   h_hat_diff = Vector{Float64}(undef, T)

   @inbounds for t in T-1:-1:1
       h_hat_diff[t] = (v[t]/F[t]) + (phi - phi*K[t])*r 
       r = h_hat_diff[t]
    end

    # to keep it simple and high-performance, we use the standard
    # state smoother recursion:
    r_state = 0.0
    @inbounds for t in T:-1:1
        h_hat_diff[t] = (v[t]/F[t]) + (1 - K[t])*phi*r_state
        r_state = h_hat_diff[t]
    end

    # --- step 4: combine
    return h_hat_diff .+ h_plus
    
end