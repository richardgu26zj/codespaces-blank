using Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote

# Objective: a noisy quadratic function with many local ripples
# We want to find the bottom of the bowl, despite the wiggles (sin term)
# f0 is Float32 version of 0.0, which is faster for optimization and commonly used in ML
objective(x, p) = sum(abs2, x .- 5.0f0) + 0.5f0*sum(sin.(5.0f0 .* x))

x0 = [0.0f0, 0.0f0]  # Starting far away at (0,0)
# this line creates OptimizationFunction obect
# the objective points to the function f(x, p) defined above
# AutoZygote() tells Optimization.jl to use Zygote whenver the optimizer needs gradients (the slope)
func = OptimizationFunction(objective, Optimization.AutoZygote())
# it defines the "starting point" of the optimization problem
prob = OptimizationProblem(func, x0)

# Adam requires a learning rate, 0.01 is a safe starting point
# It takes many small steps to ignore the "sin" wiggles
sol = solve(prob, Adam(0.01f0), maxiters=1000)

println("Solution found: ", sol.u)


#################################
# objective: The Rosenbrock function
rosenbrock(x, p) = (1.0f0 - x[1])^2 + 100.0f0 * (x[2] - x[1]^2)^2

x0 = [-1.2f0, 1.0f0]  # A common starting point for Rosenbrock
func = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
prob = OptimizationProblem(func, x0)

# BFGS does not need a learning rate, it estimates curvature internally.
sol = solve(prob, BFGS())

println("Rosenbrock solution found: ", sol.u)

#################################
# objective: high-dimensional sphere function
target_grid = collect(range(1.0f0, 2.0f0, length=50))
f(x, p) = sum(abs2, x .- p)

x0 = zeros(Float32, 50)  # 50-dimensional starting point at the origin
optf = OptimizationFunction(f, Optimization.AutoZygote())
prob = OptimizationProblem(optf, x0, target_grid)

# Using LBFGS for high-dimensional optimization
# L is limited memory, so it uses less RAM for big models
sol = solve(prob, LBFGS())

println("High-dimensional solution found: ", sol.objective)

