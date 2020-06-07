using UnicodePlots


# a very simple one hidden layer seq2one model
model = irnn(2,1,relu)
param = paramsof(model)
model.w.value[1] = 1.0
model.w.value[2] = 2.0
model.u.value[1] = 1.0

epoch = 10
lossv = zeros(epoch,1)
for e = 1:epoch
    # time step one
    x1 = Variable(ones(2,1))
    y1 = forward(model,x1)
    # time step two
    x2 = Variable(ones(2,1))
    y2 = forward(model,x2)
    # loss calculation + backward propagation + update parameters
    loss = mseLoss(y2,Variable(5.0*ones(1,1)))
    backward()
    update(param,0.005)
    zerograds(param)
    # reset hidden state is an optional choice
    resethidden(model)
    println(loss.value)
    lossv[e] = loss.value
end
lineplot(vec(lossv),xlabel="epoch", ylabel="Loss",margin=3)


# a very simple two hidden layers seq2one model
topology = [2, 32, 16, 1]
operator = [relu, relu, leakyrelu]
rnnmodel = IRNN(topology, operator)

param = paramsof(rnnmodel)
epoch = 10
lossv = zeros(epoch,1)
for e = 1:epoch
    x1 = Variable(ones(2,1))
    y1 = forward(rnnmodel,x1)
    x2 = Variable(ones(2,1))
    y2 = forward(rnnmodel,x1)
    loss = mseLoss(y2,Variable(5.0*ones(1,1)))
    backward()
    update(param,0.005)
    zerograds(param)
    resethidden(rnnmodel)
    println(loss.value)
    lossv[e] = loss.value
end
lineplot(vec(lossv),xlabel="epoch", ylabel="Loss",margin=3)
