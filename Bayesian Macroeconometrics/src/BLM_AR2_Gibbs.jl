

function BLM_AR2_Gibbs(y, X, prior; niter::Float64, nburn::Float64)
    @unpack beta0, iVbeta, nu0, S0 = prior

    T, k = size(X)
    store_param = zeros(niter, k + 1)

    # calculate a few things before Gibbs loop
    XX = X' * X
    Xy = X' * y 

    # initial values
    beta = XX\Xy
    e = y - X * beta
    sig2 = (e'*e) / (T - k)

    

