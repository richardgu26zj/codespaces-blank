using LinearAlgebra, Statistics, Distributions

struct OLSResults{T <: AbstractFloat}
    meth::String        # Method
    beta::Vector{T}     # coefficients
    tstat::Vector{T}    # t Statistics 
    bstd::Vector{T}
    p_values::Vector{T}
    yhat::Vector{T}
    resid::Vector{T}
    sige::T
    rsqr::T
    rbar::T
    dw::T
    nobs::Int 
    nvar::Int
    y::Vector{T}
    bint::Matrix{T}     # beta confident interval
end


function ols(y::AbstractVector{T}, 
    x::AbstractMatrix{T}) where T<:AbstractFloat
    nobs, nvar = size(x)
    @assert nobs == length(y) "Dimension mismatch!"

    # 1. efficiently solve for beta 
    beta = x\y 

    # 2. residuals and Sigma
    yhat = x*beta
    resid = y - yhat
    sigu = dot(resid, resid)
    sige = sigu/(nobs - nvar)

    # 3. covariance matrix using Cholesky on 
    # Gram matrix (X'X)
    C = cholesky(Hermitian(x'*x))
    xpxi = C\I(nvar)
    sigb = sqrt.(sige .*diag(xpxi))

    # 4. inference
    tdist = TDist(nobs - nvar)
    tcrit = quantile(tdist, 0.975)   

    tstat = beta ./sigb
    p_values = 2 .*(1 .-cdf.(tdist, abs.(tstat)))
    bint = [beta .- tcrit .*sigb  beta .+ tcrit .*sigb]

    # 5. goodness of fit 
    y_mean_sub = y .- mean(y)
    rsqr2 = dot(y_mean_sub, y_mean_sub)
    rsqr = 1.0 - (sigu/rsqr2)
    rbar = 1.0 - (1.0 - rsqr)*(nobs-1)/(nobs-nvar)

    #6. Durbin-Watson
    res_diff = @views resid[2:end] .- resid[1:end-1]
    dw = dot(res_diff, res_diff)/sigu

    return OLSResults("ols", beta, tstat, sigb, p_values, yhat,
    resid, sige, rsqr, rbar, dw, nobs, nvar, y, bint)
end

# --- Generate Synthetic Data ---

n = 1000;
# Add a column of 1s for the intercept
X = hcat(ones(n), randn(n, 2)) ;
# True betas: Intercept=1.5, x1=2.0, x2=-0.5
true_beta = [1.5, 2.0, -0.5];
# Generate y with some noise
y = X * true_beta + randn(n) * 5;

results = ols(y,X);

println("Estimated Betas: ", round.(results.beta, digits=3))
println("Estimated Std of Betas:  ", round.(results.bstd, digits=3))
println("p_values of Betas: ", round.(results.p_values, digits=4))
println("R-Squared: ", round(results.rsqr, digits=4))
println("95% Confidence Intervals:\n", round.(results.bint, digits=3))

results.dw