using LinearAlgebra
```
The state space is given by
y_t = Hs_t
s_{t+1} = Gs_t + M*e_t

```
function kf_svd(y :: AbstractVector, H :: AbstractMatrix, shat :: AbstractVector,
                         sig :: AbstractMatrix, G :: AbstractMatrix, M :: AbstractMatrix;
                         tol :: Float64 = 1e-12)
    # 1. state covariance prediction (omega)
    omega = Symmetric(G*sig*G' + M*M')

    # 2. SVD of omega
    # Julia provides SVD in struct
    F0 = svd(omega)
    u0, d0_vals, v0 = F0.U, F0.S, F0.V

    # 3. project state uncertainty into observation space
    A = H*u0*Diagonal(sqrt.(d0_vals))
    F = svd(A)
    u, d_vals, v = F.U, F.S, F.V

    # 4. handle singuarity (truncation)
    n_valid = count(>(tol), d_vals)

    # Views avoid allocating new sub-matrices
    u_q = view(u, :, 1:n_valid)
    v_q = view(v, :, 1:n_valid)
    d_q = view(d_vals, 1:n_valid)

    # 5. forecast error 
    fac = v0*Diagonal(sqrt.(d0_vals))
    yhat = y - H*G*shat

    # ferr = v*inv(d)*u'*yhat
    ferr = v_q*(u_q'*yhat ./d_q)

    # 6. log-likelihood
    lh = [-0.5*dot(ferr, ferr), -2*sum(log, d_q)-(length(y)/2)*log(2π)]

    # 7. state update
    shatnew = G*shat + fac*ferr

    # 8. covariance update
    signew = fac*(I - v_q*v_q')*fac'

    return shatnew, signew, lh, yhat

end


