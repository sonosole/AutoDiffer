# [0] prepare model
ichannels = 32
ochannels = 1
c1 = conv_1xd(ichannels,4,3)
c2 = conv_1xd(4,ochannels,2)

timeSteps = 128
batchsize = 1

# [1] prepare input data and its label
x  = Variable(rand(ichannels,timeSteps,batchsize),true)
l  = Variable(rand(ochannels,125,batchsize),true)

# [2] forward and backward propagation
o1 = forward(c1,  x)
o2 = forward(c2, o1)
LOSS1 = mseLoss(sigmoid(o2), l)
backward()
GRAD = c1.w.delta[1,1]

# [3] forward and backward propagation with a samll change of a weight
DELTA = 1e-7
c1.w.value[1,1] += DELTA

time1 = time()
o1 = forward(c1,  x)
o2 = forward(c2, o1)
LOSS2 = mseLoss(sigmoid(o2), l)
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
