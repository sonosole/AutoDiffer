using UnicodePlots

# define a simple model of several kinds of network
cmodel = Chain([dense(2,16),dropout(0.09),dense(16,16),irnn(16,1,leakyrelu)])
# or equivalently construct a model like this:
cmodel = Chain(dense(2,16),dropout(0.09),dense(16,16),irnn(16,1,leakyrelu))
params = paramsof(cmodel)

epoch = 30
lossv = zeros(epoch,1)
for e = 1:epoch
    x1 = Variable(ones(2,1))
    y1 = forward(cmodel,x1)
    x2 = Variable(ones(2,1))
    y2 = forward(cmodel,x2)
    loss = mseLoss(y2,Variable(5.0*ones(1,1)))
    backward()
    update(params,0.01)
    zerograds(params)
    println(loss.value)
    lossv[e] = loss.value
end
lineplot(vec(lossv),xlabel="epoch", ylabel="Loss",margin=3)
