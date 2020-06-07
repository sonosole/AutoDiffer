# [1] prepare input data and its label
x = randn(256, 62)
l = rand(64, 62)
l = l ./ sum(l,dims=1)
m = MLP([256, 128,128,128,128,128, 64], [linearsin,sin,tanh,cos,sigmoid,softmax])

# [2] forward and backward propagation
outs = forward(m, Variable(x))
loss = crossEntropy(outs, Variable(l))
LOSS1 = cost(loss)
backward()
GRAD = m.layers[1].w.delta[1,1]
println("\n---------------------------------")
println("反向传播梯度: ", GRAD)


# [3] forward and backward propagation with a samll change of a weight
DELTA = 1e-6
m.layers[1].w.value[1,1] += DELTA

time1 = time()
outs = forward(m, Variable(x))
loss = crossEntropy(outs, Variable(l))
LOSS2 = cost(loss)
time2 = time()
backward()
time3 = time()

# [4] check if the auto-grad is true or not
dLdW = (LOSS2.value - LOSS1.value)/DELTA
err  = abs((dLdW-GRAD)/GRAD)*100
err  = err < 1e-5 ? 0.0 : err
println("数值计算梯度: ", dLdW)
println("相对梯度误差：", err," %")

println("\n----------- 耗时(ms) ------------")
println(" forward time: ",(time2-time1)*1000.,"\nbackward time: ", (time3-time2)*1000.)
println("---------------------------------")

# end for gradient checking
