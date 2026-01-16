using CSV, DataFrames,LinearAlgebra, Random, Distributions, Parameters, Plots
using Revise, ProgressMeter,InteractiveUtils,StaticArrays

includet("GibbsSVprior.jl")
includet("SVRW_mod.jl")
includet("Gibbs_SV_run.jl")
#includet("GibbsSVrun_main.jl")

# load Data
rawdata = CSV.read("data/PCE.csv",DataFrame)
data = Vector(rawdata[1:end-1,2])
y = 400 .*log.(data[2:end]./data[1:end-1])

@time store_h, store_para = Gibbs_SV_run(y,GibbsSVprior())
#@time store_h, store_para = GibbsSVrun_main(y,GibbsSVprior())
#p = GibbsSVprior()
#@code_warntype Gibbs_SV_run(y,p)

using Statistics

h_mode = mean(store_h,dims=2)
T = length(h_mode)
h_lower = [@views quantile(store_h[t,:],0.05) for t in 1:T]
h_upper = [@views quantile(store_h[t,:],0.95) for t in 1:T]

# Create the plot
plot(h_mode, 
    linecolor = :blue, 
    linewidth = 2.5, 
    label = "Posterior Mean", 
    title = "Stochastic Volatility (90% Credible Interval)",
    ylabel = "Volatility", 
    xlabel = "Time",
    legend = :topright)

# Add the shaded 90% interval
plot!(h_lower, 
    fillrange = h_upper, 
    fillalpha = 0.3, 
    fillcolor = :grey, 
    linealpha = 0, 
    label = "90% Credible Interval")


