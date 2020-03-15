include("../src/mlp.jl")
using Gadfly

# first class data and lable
x1 = 3*randn(2,2000);
label1 = zeros(2,2000);
label1[1,:] .= 1.0;

# second class data and lable
t = 0:pi/1000:2*pi;
x2 = zeros(2,length(t));
label2 = zeros(2,length(t));
label2[2,:] .= 1.0;
for i = 1:length(t)
    r = randn(1,1);
    r = r .+ 17.0;
    x2[1,i] = r[1].*cos(t[i]);
    x2[2,i] = r[1].*sin(t[i]);
end


input = hcat(x1,x2)
label = hcat(label1,label2)

topology = [2, 32,16,8, 2]
operator = [relu, relu, swish, softmax]
mlpmodel = MLP(topology, operator)
params   = paramsof(mlpmodel)

epoch = 1500
lrate = 2e-6
lossval = zeros(epoch,1)
tic = time()
for i=1:epoch
    outs = forward(mlpmodel, Variable(input))
    COST = mseLoss(outs, Variable(label))
    zerograds(mlpmodel)
    backward()
    update(params, lrate)
    lossval[i] = COST.value
end
toc = time()
println("\n time: ", toc-tic," sec")
println(" loss: ", lossval[end])

# a predicting example
out1 = predicate(mlpmodel, x1 .+ 1e-4)
out2 = predicate(mlpmodel, x2 .+ 1e-4)

p0 = plot(y=(lossval .+ 1e-100), Geom.line);
p1 = plot(
layer(x=out1[1,:],y=out1[2,:],Theme(default_color=colorant"blue")),
layer(x=out2[1,:],y=out2[2,:],Theme(default_color=colorant"red")))
vstack(p0,p1)
