using CSV, DataFrames, Plots, Plots.Measures, Random, Distributions
using ProgressMeter,LinearAlgebra, Parameters


includet("BLM_AR2_Gibbs.jl")

# Load Data
rawdata = CSV.read("data/PCE.csv",DataFrame);
data = Vector(rawdata[1:end-1,2]);
data = 400 .*log.(data[2:end]./data[1:end-1]);
y = data[3:end];
T = length(y);
X = [ones(T,1) data[2:end-1] data[1:end-2]];
k = size(X,2);
date = range(1959.25, stop = 2025.25, length = length(y));

beta0 = zeros(k)
iVbeta = I(k)/100
nu0 = 3.0
S0 = 0.01*(nu0-1)

prior = (; beta0, iVbeta, nu0, S0)

niter = 500_000
nburn = 10_000

@time store_param = BLM_AR2_Gibbs(y,X,prior,niter,nburn);

#plotting posterior distributions
gr()
p1 = histogram(store_param[:,1], 
    bins = 50,
    normalize = :pdf,
    color = :lightblue,
    legend = false,
    xlabel = "Coefficient on Constant",
    ylabel = "Density",
    title = "Posterior Distribution of Constant Term",
    titlefont = font(12,"Computer Modern"),
    guidefont = font(10, "Computer Modern"),
    tickfont = font(8, "Computer Modern")
    )
p2 = histogram(store_param[:,2], 
    bins = 50,      
    normalize = :pdf,
    color = :lightblue,    
    legend = false,
    xlabel = "Coefficient on Lag 1",
    ylabel = "Density",     
    title = "Posterior Distribution of Lag 1 Coefficient",
    titlefont = font(12,"Computer Modern"),
    guidefont = font(10, "Computer Modern"),
    tickfont = font(8, "Computer Modern")
    )   

p3 = histogram(store_param[:,3],
    bins = 50,      
    normalize = :pdf,
    color = :lightblue,    
    legend = false,
    xlabel = "Coefficient on Lag 2",
    ylabel = "Density",     
    title = "Posterior Distribution of Lag 2 Coefficient",
    titlefont = font(12,"Computer Modern"),
    guidefont = font(10, "Computer Modern"),
    tickfont = font(8, "Computer Modern")
    )
p4 = histogram(store_param[:,4],
    bins = 50,  
    normalize = :pdf,
    color = :lightblue,
    legend = false,
    xlabel = "Variance of Error Term",
    ylabel = "Density",     
    title = "Posterior Distribution of Error Variance", 
    titlefont = font(12,"Computer Modern"),
    guidefont = font(10, "Computer Modern"),
    tickfont = font(8, "Computer Modern")
    )  

plot(p1, p2, p3, p4, layout = (2,2), size = (1000,600))

savefig("figures/BLM_AR2_posteriors.pdf")
