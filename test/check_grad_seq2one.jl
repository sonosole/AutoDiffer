rnnmodel = Chain(
dense(2,128,sin),
dense(128,128,tanh),
residual(dense(128,64),linear(64,128)),
indrnn(128,64),
indlstm(64,64),
dense(64,1)
)
resethidden(rnnmodel)

for t = 1:20
    global y = forward(rnnmodel,Variable(ones(2,1)))
end
LOSS1 = mseLoss(y,Variable(5.0*ones(1,1)))
backward()

DELTA = 1e-6
GRAD  = rnnmodel.blocks[1].w.delta[1,1]
rnnmodel.blocks[1].w.value[1,1] += DELTA
resethidden(rnnmodel)
for t = 1:20
    global y = forward(rnnmodel,Variable(ones(2,1)))
end
LOSS2 = mseLoss(y,Variable(5.0*ones(1,1)))


dLdW = (LOSS2.value - LOSS1.value)/DELTA
err  = abs((dLdW-GRAD)/(GRAD+1e-38))*100
err  = err < 1e-3 ? 0.0 : err


println("\n---------------------------------")
println("backward  gradient: ", round(GRAD, digits=9))
println("numerical gradient: ", round(dLdW, digits=9))
println("( relative error ): ", round(err,  digits=7)," %")
println("----------------------------------")
