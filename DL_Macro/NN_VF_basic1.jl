using Lux, Random

# 1. define the model
# One asset 'a' as input, 16 hidden nodes, one Value 'V' as output
v_func_model = Chain(
    Dense(1, 16, Lux.tanh), # intput (1), neurons (16): the width of hidden layer
    Dense(16, 1), # output(1): the scalar value of the value function V(a) at that point
)
# print the keys and sizes of the parameters
println("Available fields in ps:   ", keys(ps))
println("Layer 1 weights: ", size(ps.layer_1.weight))
println("Layer 1 bias: ", size(ps.layer_1.bias))

# 2. initialize the actual numbers (parameters & state)
rng = Random.default_rng()
ps, st = Lux.setup(rng, v_func_model)

# 3. implementation: output = model(intput, parameters, state)
a_sample = randn(Float32, 1, 10) # 10 random asset points
V_pred, st_new  = v_func_model(a_sample, ps, st)

println(V_pred) # print the predicted values