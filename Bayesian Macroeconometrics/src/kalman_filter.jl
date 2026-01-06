using LinearAlgebra

function kalman_filter(
	A 		  	:: AbstractMatrix{FT},  # transition matrix
	Q 		  	:: AbstractMatrix{FT},  # state transition covariance
	H           :: AbstractMatrix{FT},  # measure matrix
	R           :: AbstractMatrix{FT},  # measure covariance
	S_prev   :: AbstractVector{FT},  # state estimate (t-1)
	P_prev   :: AbstractMatrix{FT},  # state covariance (t-1)
	y		     :: AbstractVector{FT}  # observations
	) where {FT <: AbstractFloat}

	# 1. state prediction
	S_pred = A * S_prev
	P_pred = A *P_prev * A' + Q

	# 2. forecast
	y_pred = H * S_pred
	F_pred = H * P_pred * H' + R 

	# 3. Innovation 
	v = y - y_pred

	# 4. log-likelihood (using Cholesky for stability and speed)
	# F_pred must be positive definite
	tmp = cholesky(Hermitian(F_pred))

	# log(def(F)) = 2*sum(log(diag(L)))
	# the trick is that tmp\v = x = (L')^{-1}L^{-1} v = (LL')^{-1}v = F^{-1} v
	# it is not one operation, but two together
	loglik = -0.5*(logdet(tmp) + v'*(tmp\v) + length(v)*log(2*pi))

	# 5. Kalman gain
	# solving F_pred'*K' = (P_pred*H')' is more efficient than inverse operation
	K = (P_pred*H')/tmp

	# 6. Update (Joseph form)
	I_KH = I - K*H
	S_update = S_pred + K*v
	P_update = I_KH*P_pred*I_KH' + K*R*K'

	return S_update, P_update, loglik

end
