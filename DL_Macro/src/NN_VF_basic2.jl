using Lux, Random, Zygote

# Define a simple neural network model
v_func_model = Chain(
    Dense(1, 16, Lux.tanh),
    Dense(16, 1),
)
rng = Random.default_rng()
ps, st = Lux.setup(rng, v_func_model)
a_inputs = randn(Float32, 1, 10)

# step 4: calculating the derivative V'(a)
v_only(x) = v_func_model(x, ps, st)[1] # function that returns only V not state

# calculate the Jacobian for all 10 points at once
@time dV_da = Zygote.jacobian(v_only, a_inputs)[1]
@time dV_da2 = Zygote.gradient(x -> sum(v_func_model(x, ps, st)[1]), a_inputs)[1]
#dV_da3 = Lux.batched_jacobian(v_func_model, a_inputs, ps, st)[1]

# dV_da will be a 1x10 matrix, where each element corresponds to the derivative of V with respect to a at each input point
println("Derivatives of V with respect to a at each input point:", dV_da2)