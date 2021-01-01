# [1] prepare input data and its pooling label
x = randn(256, 62)
l = rand(64, 1)
l = l ./ sum(l,dims=1)

blocks =
[
    dense(256,128,relu),
    maxout(128,128,k=3),
    dense(128,128,tanh),
    residual(dense(128,64,sin),linear(64,128)),
    dense(128,128,cos),
    maxout(128,128,k=2),
    dense(128,64,sigmoid)
]

m = Chain(blocks)

# [2] forward and backward propagation
outs = forward(m, Variable(x))
loss = crossEntropy(linearpool(outs), Variable(l))
LOSS1 = cost(loss)
backward()
GRAD = blocks[1].w.delta[1,1]

# [3] forward and backward propagation with a samll change of a weight
DELTA = 1e-6
blocks[1].w.value[1,1] += DELTA

time1 = time()
outs = forward(m, Variable(x))
loss = crossEntropy(linearpool(outs), Variable(l))
LOSS2 = cost(loss)
time2 = time()
backward()
time3 = time()

# [4] check if the auto-grad is true or not
dLdW = (LOSS2.value - LOSS1.value)/DELTA
err  = abs((dLdW-GRAD)/(GRAD+1e-38))*100
err  = err < 1e-3 ? 0.0 : err

println("\n---------------------------------")
println("backward  gradient: ", round(GRAD, digits=9))
println("numerical gradient: ", round(dLdW, digits=9))
println("( relative error ): ", round(err,  digits=7)," %")
println("\n-------- time spent (ms) ---------")
println(" forward  time: ",(time2-time1)*1000.)
println(" backward time: ",(time3-time2)*1000.)
println("----------------------------------")
# end for gradient checking
