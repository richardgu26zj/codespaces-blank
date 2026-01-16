using LinearAlgebra, Random, Distributions, Parameters
using Parameters:unpack

function AR3_Gibbs_sampler(y::Vector{Float64}, X::Matrix{Float64},
                           prior::AR3Bayesprior;
                           Nsim::Int64=50_000, Nburn::Int64=5_000)
    @unpack beta0, iVbeta, nu0, S0 = prior;

    # calculate a few things for MCMC loop
    T, k = size(X);
    XX   = X'*X;
    Xy   = X'*y;
    iVbeta_beta0 = iVbeta*beta0; 

    # initialize storage
    resid = Vector{Float64}(undef,T);
    store_param = zeros(Nsim, k+1);

    # initialize MCMC
    beta = XX\Xy;
    #sig2 = (y - X*beta)'*(y - X*beta)/T;
    sig2 = dot(y - X*beta, y - X*beta)/T;

    Random.seed!(124); # for repoducibility

    println("MCMC of Bayesian AR(3) Model Starts here...\n")

    for isim in 1:(Nsim + Nburn)
        # 1. sample beta from Gaussian 
        Dbeta    = Symmetric(XX./sig2 + iVbeta);
        C        = cholesky(Dbeta); 
        #beta_hat = Dbeta\(Xy/sig2 + iVbeta*beta0);
        beta_hat = C\(Xy./sig2 + iVbeta_beta0);
        beta     = beta_hat + C.L'\randn(k);

        # 2. sample sig2 from Inverse-Gamma 
        # tmp  = y - X*beta;
        # SSR  = dot(tmp, tmp);
        copyto!(resid,y);
        mul!(resid,X,beta,-1.0,1.0); # resid = 1.0*y-1.0*X*beta
        SSR = dot(resid,resid);
        sig2 = rand(InverseGamma(nu0+T/2, S0+SSR/2));

        if isim > Nburn
            isave = isim - Nburn;
            #store_param[isave,:] = [beta' sig2];
            store_param[isave,1:k] .= vec(beta); #.= copy the element directly
            # into pre-allocated row of store_param one by one
            store_param[isave,k+1]  = sig2;
        end # end of storage

    end # end of MCMC loop 

    println("MCMC loop complete!\n")

    return store_param

end



