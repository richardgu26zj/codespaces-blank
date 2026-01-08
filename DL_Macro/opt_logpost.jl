using Optimization, OptimizationOptimJL, ForwardDiff, LinearAlgebra

# 1. define the data and log-posterior
data = [1.2, 0.9, 1.1, 1.3, 0.8]

function objective(u, p)
    μ = u[1]
    # log-likelihood Normal
    log_lik = sum( - 0.5 .*(data .-μ).^2 )
    # log-prior Normal(0, 10)
    log_prior = -0.5*(μ/10)^2
    return - (log_lik + log_prior)  # negative log-posterior
end

# 2. Set up the Optimization problem with Automatic Differentiation
# autoforwarddiff() tells the package to handle gradients and Hessian for us
opt_func = OptimizationFunction(objective, Optimization.AutoForwardDiff())

u0 = [0.0]  # initial guess for μ
prob = OptimizationProblem(opt_func, u0)

# 3. Solve the optimization problem using BFGS
sol = solve(prob, BFGS())

# 4. extract and print the results
mode = sol.u[1]
hessian_val = ForwardDiff.hessian(u -> objective(u, nothing), sol.u)

println("MAP Estimate (Mode):  ", mode)
println("Hessian at Mode: ", hessian_val[1,1])
println("Standard Deviation at Mode: ", sqrt(1 / hessian_val[1,1]))
