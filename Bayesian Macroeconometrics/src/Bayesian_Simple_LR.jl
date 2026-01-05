using CSV, DataFrames, Plots, Plots.Measures, Random, Distributions, ProgressMeter,LinearAlgebra

# using Pkg
# Pkg.activate(@__DIR__)
# cd(@__DIR__)

# Load Data
rawdata = CSV.read("data/PCE.csv",DataFrame);
data = Vector(rawdata[1:end-1,2]);
data = 400 .*log.(data[2:end]./data[1:end-1]);
y = data[3:end];
T = length(y);
X = [ones(T,1) data[2:end-1] data[1:end-2]];
k = size(X,2);
date = range(1959.25, stop = 2025.25, length = length(y));

# Initialize plotting of data
# gr()
# plot(date, data, 
#      color =:blue,
#      lw = 2.5, 
#      grid =:both,
#      gridalpha = 0.15,
#      legend = false,
#      xlabel = "Quarters (Q)",
#      ylabel = "Inflation Rate (%)",
#      guidefont = font(10, "Computer Modern"),
#      title = "US Inflation Rate (PCE)",
#      titlefont = font(16,"Computer Modern"),
#      xlims = (1959.00, 2025.50),
#      tickfont = font(8, "Computer Modern")
#      )

# savefig("figures/inflation.pdf")


# MCMC setup 
Nsim = 50000; 
Nburn = 5000;
function LM_AR2_Gibbs(y,X,Nsim,Nburn)
    # setup
    T,k = size(X);
    store_param = zeros(Nsim,k+1);

    # prior setup
    beta0 = zeros(k,1); iVbeta = I(k)/100;
    nu0 = 3.0; S0 = 1*(nu0-1);
    XX = X'*X; 
    Xy = X'*y;
    beta = XX\Xy;
    sig2 = sum(abs2, y - X*beta)/T;

    Random.seed!(123)

    @showprogress for isim  in 1:(Nsim + Nburn)
        #--- sample beta -----
        Dbeta = XX/sig2 + iVbeta; 
        beta_hat = Dbeta\(Xy/sig2 + iVbeta*beta0);
        L = cholesky(Hermitian(Dbeta)).L;
        beta = beta_hat + L'\randn(k,1);

        # --- sample sig2 ----
        ssr = sum(abs2,y-X*beta);
        sig2 = rand(InverseGamma(nu0+T/2, S0 + ssr/2));

        if isim > Nburn
            isave = isim - Nburn;
            store_param[isave,:] = [beta' sig2];
        end

    end
    return store_param
end

@time store_param = LM_AR2_Gibbs(y,X,Nsim,Nburn)

# plotting
gr()
labels = ["intercept","lag 1 coefficient",
        "lag 2 coefficient","variance"];

plots = [histogram(store_param[:,i],
        bins = :auto,
        color = :steelblue,
        alpha = 0.7,
        title = labels[i],
        legend = false,
        titlefont = font(10, "serif", :bold)) for i in 1:(k+1)];

plot(plots..., layout = (2,2), size = (950, 600))

savefig("figures/LM_AR2_hist_plot.pdf");



