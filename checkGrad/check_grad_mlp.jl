include("../src/mlp.jl")

# [1] prepare input data and its label
x = randn(256, 62)
l = rand(10, 62)
l = l ./ sum(l,dims=1)
m = MLP([256, 128,32,16, 10], [relu,relu,SIN,softmax])

# [2] forward and backward propagation
g = Graph(true)
o, params = forward(g, m, x)
loss = crossEntropy(g, o, Variable(l))
# loss = mse(g, o, Variable(l))

LOSS1 = cost(g, loss)
Backward(g)

GRAD = params[1].delta[1,1]
println("\n---------------------------")
println("反向传播梯度: ", GRAD)


# [3] forward and backward propagation with a samll change of a weight
DELTA = 1e-7
m.parameter[1].w[1,1] += DELTA

time1 = time()
g = Graph(true)
o, params = forward(g, m, x)
loss = crossEntropy(g, o, Variable(l))
# loss = mse(g, o, Variable(l))
LOSS2 = cost(g, loss)
time2 = time()
Backward(g)
time3 = time()

# [4] check if the auto-grad is true or not
dLdW = (LOSS2.value-LOSS1.value)/DELTA
err  = abs((dLdW-GRAD)/GRAD)
err  = err < 1/100000 ? 0.0 : err
println("数值计算梯度: ", dLdW)
println("相对梯度误差：", err," %")

println("\n--------- 耗时(ms) -----------")
println(" forward: ",(time2-time1)*1000.,"\nbackward: ", (time3-time2)*1000.)
println("------------------------------")

# end for gradient checking