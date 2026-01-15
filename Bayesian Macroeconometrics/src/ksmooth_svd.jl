

function ksmooth_svd(shats :: AbstractVector, sigs :: AbstractMatrix, G :: AbstractMatrix)
    T = length(shats)
    shat_smoothed = copy(shats)

    # backward recursion (T-1 down to 1)
    for t in (T-1):-1:1
        # prediction for t+1 based on t
        P_pred = G*sigs[t]*G' + M*M';

        # kalman smoother gain
        J = sigs[t]*G'/P_pred

        shat_smoothed[t] += J*(shat_smoothed[t+1] - G*shats[t])
    end

    return shat_smoothed
end


