using LinearAlgebra, Statistics, Parameters,Dates,Printf

# load data
rawdata = CSV.read("data/PCE.csv",DataFrame);
data = Vector(rawdata[1:end-1,2]);
data = 400 .*diff(log.(data));
y = data[4:end]
T = length(y)
X = [ones(T,1) data[3:end-1] data[2:end-2] data[1:end-3]]
T,k = size(X)

# ---THE INPUT STRUCT---
@with_kw struct AR3Bayesprior
    # Model Metadata
    name::String = "Bayesian AR(3) Model";
    timestamp::String = Dates.format(now(),"yyyy-mm-dd HH:MM");

    # Priors
    k::Int = 4; # intercept + 3 lags
    beta0::Vector{Float64} = zeros(k)
    iVbeta::Matrix{Float64} = 0.01*I(k);
    nu0::Float64 = 3.0; 
    S0::Float64 = 1.0*(nu0-1);
end

# --- THE OUTPUT STRUCT --- 
@with_kw struct AR3BayesResults
    # link back to the settings used
    prior::AR3Bayesprior

    # Summary Statistics
    beta_mode::Vector{Float64}
    beta_std::Vector{Float64}
    sig2::Float64

    # Raw Draws(The posterior)
    postdraws::Matrix{Float64}

    # Computation Metadata
    runtime_seconds::Float64
end

includet("AR3_Gibbs_sampler.jl")

function run_estimation(y,X,prior::AR3Bayesprior)
    # start timer
    time_start = time()

    # Run the Gibbs sampler 
    raw_draws = AR3_Gibbs_sampler(y,X,prior)

    # calculate summary statistics from the draws 
    b_mode = vec(mean(raw_draws[:, 1:prior.k], dims = 1))
    b_std  = vec(std(raw_draws[:,1:prior.k],dims=1))
    s_mean = mean(raw_draws[:,end])

    time_end = time()

    # wrap everything into the results struct
    return AR3BayesResults(
        prior = prior,
        beta_mode = b_mode,
        beta_std = b_std,
        sig2 = s_mean,
        postdraws = raw_draws,
        runtime_seconds = time_end - time_start
    )
end

p1 = AR3Bayesprior(name = "Baseline")
p2 = AR3Bayesprior(name = "Tight Prior", iVbeta = 1.0*I(k))
p3 = AR3Bayesprior(name="Theory Mean", beta0=[1.0, 0.5, 0.2, 0.1])

#raw_draw = AR3_Gibbs_sampler(y,X,p1)
# Run Estimation 
#res1 = run_estimation(y,X,p1)
#res2 = run_estimation(y,X,p2)
results_list = [run_estimation(y,X,p) for p in [p1, p2, p3]]

# Results Comparison
# println("Model comparison:\n")
# @printf("%-20s %-15s %-15s\n","Metric",res1.prior.name,res2.prior.name)
# @printf("%-20s %-15.4f %-15.4f\n","Posterior Mean Sig2",res1.sig2, res2.sig2)

for res in results_list
    println("Results for: ", res.prior.name)
    println("Beta mean:   ", res.beta_mode)
    println("Sigma:", res.sig2)
    println("Run time:", res.runtime_seconds)
    println("-"^20)
end


using StatsPlots

density([res.postdraws[:,2] for res in results_list],
        label = ["Baseline" "Tight Prior" "Theory Mean"],
        title = "Posterior Density of First-Lag Coeffcient",
        lw = 3)

        # 1. Extract the data (this creates a Vector of Vectors)
plot_data = [res.postdraws[:, 2] for res in results_list]

# 2. Plot with explicit "Row" labels and alpha transparency
density(plot_data, 
        label = ["Baseline" "Tight Prior" "Theory Mean"], # NO COMMAS between labels
        fill = (0, 0.2),      # Adds a light tint under the curves
        linewidth = 2.5, 
        title = "Posterior Comparison",
        xlabel = "Coefficient Value",
        ylabel = "Density")

println("Hypothesis Testing: P(Beta_1 > 0)")
for res in results_list
    prob = mean(res.postdraws[:, 2] .> 0)
    @printf("%-15s: %6.2f%%\n", res.prior.name, prob * 100)
end

println("-"^50)
@printf("%-10s %-10s %-25s\n", "Variable", "Mean", "95% Credible Interval")
println("-"^50)

# k is the number of betas
for i in 1:res1.prior.k
    b_mean = res1.beta_mode[i]
    lower, upper = quantile(res1.postdraws[:, i], [0.025, 0.975])
    
    label = i == 1 ? "Intercept" : "Lag $(i-1)"
    @printf("%-10s %-10.4f [%.4f, %.4f]\n", label, b_mean, lower, upper)
end
println("-"^50)