using LinearAlgebra

```
kf(y, H, shat, sig, G, M)

Christopher Sims Matlab code of SVD-based Kalman Filter

````

function kf(y, H, shat, sig, G, M)
    # ensure vectors are column vectors (handling dimensionality)
    shat = vec(shat)
    y = vec(y)

    # 1. prediction covariance (Omega)
    omega = G*sig*G' + M*M';

    # 2. SVD of Omega
    F0 = svd(omega)
    u0, d0_vals, v0 = F0.U, F0.S, F0.V 

    # 3. project state uncertainty into observation space
    F = svd(H*u0*Diagonal(sqrt.(d0_vals)))
    u, d_vals, v = F.U, F.S, F.v

    # 4. handle singuarity (truncation)
    first0 = findfirst(x -> x<1e-12, d_vals)

    if first0 == nothing
        # keep all dimensions if no small singular values found
        idx = 1:length(d_vals)
    else
        idx = 1:(first0-1)
    end

    # slice matrices to remove singular dimensions
    u_trunc = u[:, idx]
    v_trunc = v[:, idx]
    d_inv = Diagonal(1 ./d_vals[idx])

    # 5. calculate forecast error and likelihood
    fac = v0*Diagonal(sqrt.(d0_vals))
    yhat = y - H*G*shat

    # forecast error
    ferr = v_trunc*d_inv*u_trunc'*yhat

    lh = zeros(2)
    lh[1] = -0.5*(ferr'*ferr)
    lh[2] = -sum(log.(d_vals[idx])) - (length(y)/2)*log(2*pi)

    # 6. state update
    shatnew = fac*ferr + G*shat
    n_v = size(v_trunc, 1)
    signew = fac*(I - v_trunc*d_inv*v_trunc')*fac'

    return shatnew, signew, lh, yhat

end