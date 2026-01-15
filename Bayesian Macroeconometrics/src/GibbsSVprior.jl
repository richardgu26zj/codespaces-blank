using Parameters

@with_kw struct GibbsSVprior
    # Mean and variance for h0
    ah0::Float64 = 0.1
    bh0::Float64 = 10.0

    # Mean and variance for tau (level of the process)
    atau::Float64 = 2
    btau::Float64 = 20.0

    # hyperparameters for sigh2 
    nu0::Float64 = 5.0
    S0::Float64 = 0.1*(nu0-1)

    # error check
    @assert bh0 > 0 "Variance bh0 must be positive!"
    @assert btau > 0 "Variance btau must be positive!"
end