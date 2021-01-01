timeSteps = 77
featsdims = 2
batchsize = 10

rnnmodel = Chain(
dense(featsdims,128),
indrnn(128,2),
indlstm(2,2),
dense(2,2,tanh)
)

ibatch = ones(featsdims,batchsize,timeSteps)
obatch = Vector{Variable}(undef, timeSteps)

for t = 1:timeSteps
    obatch[t] = forward(rnnmodel, Variable(ibatch[:,:,t]))
end
output = vcats(obatch)
poolvalue = linearpool(output)
LOSS1 = mseLoss(reshape(poolvalue,(featsdims,batchsize)),Variable(5*ones(2,batchsize)))
println(graph.backward|>length)
backward()

DELTA = 1e-5
GRAD  = rnnmodel.blocks[1].w.delta[1,1]
rnnmodel.blocks[1].w.value[1,1] += DELTA

resethidden(rnnmodel)
for t = 1:timeSteps
    obatch[t] = forward(rnnmodel, Variable(ibatch[:,:,t]))
end
output = vcats(obatch)
poolvalue = linearpool(output)
LOSS2 = mseLoss(reshape(poolvalue,(featsdims,batchsize)),Variable(5*ones(2,batchsize)))


dLdW = (LOSS2.value - LOSS1.value)/DELTA
err  = abs((dLdW-GRAD)/(GRAD+1e-38))*100
err  = err < 1e-3 ? 0.0 : err

println("\n---------------------------------")
println("backward  gradient: ", round(GRAD, digits=9))
println("numerical gradient: ", round(dLdW, digits=9))
println("( relative error ): ", round(err,  digits=7)," %")
println("----------------------------------")
