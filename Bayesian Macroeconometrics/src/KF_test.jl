using LinearAlgebra, Random

include("kalman_filter.jl")  # assuming the Kalman filter function is in this file

# define the stimulation parameters
Random.seed!(1234)  # for reproducibility
n_steps = 200 # number of time steps
Δt = 0.1         # time step    
σ_proc = 0.1  # process noise standard deviation
σ_meas = 0.5 # measurement noise standard deviation

# ---- Model matrices ----
A = [1.0 Δt; 
       0.0 1.0]  # state transition matrix

Q = σ_proc^2 *[Δt^3/3 Δt^2/2; 
                          Δt^2/2 Δt]  # process noise covariance

H = [1.0 0.0]  # observation matrix

R = fill(σ_meas^2, 1, 1)  # measurement noise covariance

# --- Date generation ---
true_states = Vector{Vector{Float64}}(undef, n_steps)
observations = Vector{Vector{Float64}}(undef, n_steps)

s_true = [0.0; 1.0]  # initial true state

for t  in 1:n_steps
    global s_true
    # s_t = A*s_{t-1} + w_t
    s_true = A*s_true + sqrt(Q) * randn(2)
    true_states[t] = s_true

    # y_t = H*s_t + v_t
    y_t = H*s_true + sqrt(R) * randn(1) 
    observations[t] = y_t
end

# --- Kalman Filter Implementation ---

# initial guess
S_est = [0.0; 0.0]  # initial state estimate
P_est = Matrix{Float64}(I, 2, 2)*10.0  # initial estimate covariance

estimated_states = Vector{Vector{Float64}}()
total_loglik = 0.0

@time for t in 1:n_steps
    global S_est, P_est, total_loglik
    y = observations[t]

    S_est, P_est, loglik = kalman_filter(A, Q, H, R, S_est, P_est, y)

    push!(estimated_states, S_est)
    total_loglik += loglik
end

println("Total Log-Likelihood: ", total_loglik)

# --- Results Visualization ---
using Plots     
time = collect(1:n_steps) .* Δt
true_positions = [s[1] for s in true_states]
estimated_positions = [s[1] for s in estimated_states]  
observed_positions = [y[1] for y in observations]
plot(time, true_positions, label="True Position", lw=2)
plot!(time, estimated_positions, label="Estimated Position", lw=2, ls=:dash)
scatter!(time, observed_positions, label="Observed Position", ms=3, alpha=0.5)
xlabel!("Time (s)")
ylabel!("Position")
title!("Kalman Filter Position Estimation")

savefig("figures/kalman_filter_position_estimation.pdf")




