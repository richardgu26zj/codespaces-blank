using LinearAlgebra, Random, Distributions, ProgressMeter
using Parameters: @unpack 

function BLM_AR2_Gibbs(y, X, prior, niter::Int, nburn::Int)
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

    @showprogress for iter in 1:(niter + nburn)
       
            #--- sample beta ---
            Dbeta = XX/sig2 + iVbeta;
            beta_hat = Dbeta\(Xy/sig2 + iVbeta*beta0);
            L = cholesky(Symmetric(Dbeta)).L
            beta = beta_hat + L'\randn(k);

            #--- sample sig2 ---
            e = y - X * beta
            sig2 = rand(InverseGamma(nu0 + T/2, (e'*e + S0)/2))
    
               
        #--- store parameters ---
        if iter > nburn
            isave = iter - nburn
            store_param[isave, :] = [beta' sig2]
        end
        
    end
    return store_param 
end

    
    

