using Lux, Random, Zygote, Optimization, OptimizationOptimisers, ComponentArrays, Plots

# 1. Economic Parameters
const ρ = 0.05f0
const r = 0.03f0
const w = 1.0f0
const γ = 2.0f0

# 2. Setup Model
rng = Random.default_rng()
model = Chain(Dense(1 => 16, tanh), Dense(16 => 1))
ps, st = Lux.setup(rng, model)
ps_ca = ComponentArray(ps)

# 3. Setup Grid (Fixed for Numerical Stability)
# We create a_grid and a slightly shifted grid a_plus to calculate the slope
n_points = 100
a_grid = reshape(collect(range(0.1f0, 10.0f0, length=n_points)), 1, n_points)
h = 1f-4 # The step size for the numerical derivative
a_plus = a_grid .+ h

# 4. The "No-Mutation" Loss Function
function hjb_loss(θ, p)
    # Get V(a) and V(a + h)
    # We run the model twice. Zygote loves this because it's just standard math.
    V, _ = model(a_grid, θ, st)
    V_plus, _ = model(a_plus, θ, st)
    
    # Calculate V'(a) numerically: [V(a+h) - V(a)] / h
    # This is a standard array operation. No nested Zygote calls!
    dV_da = (V_plus .- V) ./ h
    
    # Economic Logic
    # (Notice: no Zygote.gradient here, so no Mutation Error)
    dV_da_safe = max.(dV_da, 1f-6)
    c = dV_da_safe .^ (-1f0 / γ)
    
    u_c = (c .^ (1f0 - γ) .- 1f0) ./ (1f0 - γ)
    savings = (r .* a_grid) .+ w .- c
    
    # HJB Identity
    residual = (ρ .* V) .- (u_c .+ dV_da .* savings)
    
    # Total Loss
    loss_hjb = sum(abs2, residual)
    loss_monot = sum(abs2, max.(0.0f0, .-dV_da)) * 100.0f0
    
    return loss_hjb + loss_monot
end

# 5. Optimization
# We provide the objective and the AD backend for the parameters
opt_fn = OptimizationFunction((θ, p) -> hjb_loss(θ, p), Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_fn, ps_ca)

println("Starting Optimization (Numerical Derivative Mode)...")
res = solve(opt_prob, Adam(0.01f0), maxiters=1000)

# 6. Plot Results
a_plot = reshape(collect(range(0.1f0, 10.0f0, length=200)), 1, 200)
V_pred = model(a_plot, res.u, st)[1]
plot(a_plot', V_pred', title="Value Function (Numerical Grad)", xlabel="Assets", ylabel="V")


# Calculate residuals after training
V_final, _ = model(a_grid, res.u, st)
V_plus, _ = model(a_grid .+ h, res.u, st)
dV_da = (V_plus .- V_final) ./ h

# Reconstruct the HJB components
c = max.(dV_da, 1f-6) .^ (-1f0 / γ)
u_c = (c .^ (1f0 - γ) .- 1f0) ./ (1f0 - γ)
savings = (r .* a_grid) .+ w .- c
hjb_res = (ρ .* V_final) .- (u_c .+ dV_da .* savings)

# Plot the absolute error
plot(a_grid', abs.(hjb_res)', title="HJB Residual Across Assets", 
     yscale=:log10, ylabel="Absolute Error", xlabel="Assets")