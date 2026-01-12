using Lux, Zygote, Random

# Reuse the model definition and initialization from NN_VF_basic2.jl
v_func_model = Chain(
    Dense(1, 16, Lux.tanh),
    Dense(16, 1),
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, v_func_model)
a_inputs = randn(Float32, 1, 10)

v_only(x) = v_func_model(x, ps, st)[1]

dV_da = Zygote.gradient(x -> sum(v_func_model(x, ps, st)[1]), a_inputs)[1]

# 1. economic parameters
const ρ = 0.05f0       # discount rate
const r = 0.03f0       # interest rate
const w = 1.0f0        # labor income
const γ = 2.0f0       # risk aversion (for CRRA utility)

# 2. utility function
u(c) = (c ^ (1 - γ)) / (1 - γ)  # CRRA utility
inv_u_prime(v_a) = v_a ^ (-1 / γ)  # inverse of marginal utility

# 3. the residual function
function hjb_residual(ps, st, a_inputs)
    # step A: get V(a) and V'(a)
    # we use gradient(sum...) for efficiency
    V, _  = v_func_model(a_inputs, ps, st)  # V(a)
    dV_da = Zygote.gradient(x -> sum(v_func_model(x, ps, st)[1]), a_inputs)[1]  # V'(a)

    # step B: compute consumption c*
    # based on FOC: u'(c) = V'(a)  => c = inv_u_prime(V'(a))
    c_star = inv_u_prime.(abs.(dV_da) .+ 1f-6)  # ensure positivity

    # step C: compute the residual of HJB
    # ρV(a) = u(c*) + V'(a) * (r*a + w - c*)
    savings = r .*a_inputs .+ w .- c_star  # a' = r*a + w - c*
    theoretical_V = u.(c_star) .+ dV_da .* savings  # RHS of HJB
    residual = ρ .* V .- theoretical_V  # LHS - RHS
    return sum(abs2, residual)
end

# 4. test it with current parameters
loss = hjb_residual(ps, st, a_inputs)
println("HJB residual loss: ", loss) 


function calculate_euler_errors(ps, st, a_inputs)
    # 1. Get the marginal value of assets from the NN
    # This is exactly what we discussed: V'(a)
    dV_da = Zygote.gradient(x -> sum(v_func_model(x, ps, st)[1]), a_inputs)[1]
    
    # 2. Calculate optimal consumption based on our FOC: c = (V')^(-1/γ)
    # We use our 'safe' version to avoid complex numbers during training
    c_star = max.(dV_da, 1f-6) .^ (-1/γ)
    
    # 3. In a steady state HJB, the Euler Equation implies:
    # (ρ - r) * u'(c) = u''(c) * c_dot
    # However, a simpler "Unitless" way to check this is the 
    # Marginal Utility Error:
    
    # Let's check the consumption policy error:
    # We compare the NN-derived consumption to a known benchmark 
    # or check the HJB identity.
    
    # In practice, for PINNs, we often report the "Residual" we calculated earlier:
    # HJB_Error = |ρV(a) - [u(c) + V'(a)(ra + w - c)]|
    
    return abs.(ρ .* v_func_model(a_inputs, ps, st)[1] .- 
               (u.(c_star) .+ dV_da .* (r .* a_inputs .+ w .- c_star)))
end

# Evaluate after training
errors = calculate_euler_errors(res.u, st, a_inputs)
println("Average HJB/Euler Error: ", sum(errors) / length(errors))