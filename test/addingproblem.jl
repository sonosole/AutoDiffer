
rnnmodel = Chain(
indrnn(2,32),
indrnn(32,32),
indrnn(32,16),
indrnn(16,1,sigmoid)
)

params = paramsof(rnnmodel)
optimi = Momentum(params;learnRate=1e-4)

epochs = 100*100
lossval = zeros(epochs,1)
for e=1:epochs
    # T = rand(90:120,1)[1]
    x, s = addingproblemdata(100)
    resethidden(rnnmodel)
    for t = 1:T
        global y = forward(rnnmodel,Variable( reshape(x[:,t],2,1) ) )
        # println(x[:,t])
    end
    LOSS1 = mseLoss(y, Variable( reshape(s,1,1) ) )
    lossval[e] = LOSS1.value
    backward()
    update(optimi,params;clipvalue=1.0)
    zerograds(params)
end

lineplot(vec(lossval))


function addingproblemdata(T::Int)
    @assert (T>1) "The sequence length should lager than 1"
    x1 = rand(1,T)./3
    x2 = zeros(1,T)
    hf = T>>1
    I1 = rand(1:hf,1)[1]
    I2 = rand((hf+1):T,1)[1]
    # x2[I1] = 1.0
    # x2[I2] = 1.0
    x2[1] = 1.0
    x2[2] = 1.0
    y = sum(x1 .* x2)
    return [x1;x2],[y]
end
