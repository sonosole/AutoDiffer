include("../src/mlp.jl")

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
    r = r .+ 15.0;
    x2[1,i] = r[1].*cos(t[i]);
    x2[2,i] = r[1].*sin(t[i]);
end

using Gadfly
# p1 = plot(x=x1[1,:],y=x1[2,:]);
# p2 = plot(x=x2[1,:],y=x2[2,:]);
# hstack(p1, p2)

input = hcat(x1,x2)
label = hcat(label1,label2)
# plot(x=input[1,:],y=input[2,:])

topology = [2, 32, 32, 2, 2]
operator = [relu, SIN, sigmoid, softmax]
mlpmodel = MLP(topology, operator)

mlpmodel.layernum = 5
epoch = 1000
lr =  0.0001
lossval = zeros(epoch,1)
tic = time()
for i=1:epoch
    global LOSS
    g = Graph(true)
    o, w = forward(g, mlpmodel, input)
    loss = crossEntropy(g, o, Variable(label))
    LOSS = cost(g, loss)
    Backward(g)
    for i = 1:length(w)
        update(w[i], lr)
    end
    lossval[i] = LOSS.value
end
toc = time()
println(" time: ", toc-tic," sec")
println(" loss: ", LOSS.value)
p0 = plot(y=lossval, Geom.line);

# a predicting example
mlpmodel.layernum -= 1
out1 = predicate(mlpmodel, input[:,1:2000] .+ 1e-3)
out2 = predicate(mlpmodel, input[:,2001:end] .+ 1e-3)

p1 = plot(x=out1.value[1,:],y=out1.value[2,:],Coord.cartesian(xmin=0, xmax=1, ymin=0, ymax=1),Geom.histogram2d);
p2 = plot(x=out2.value[1,:],y=out2.value[2,:],Coord.cartesian(xmin=0, xmax=1, ymin=0, ymax=1),Geom.histogram2d);
h1 = hstack(p1,p2);
vstack(h1,p0)
plot(x=out2.value[1,:],y=out2.value[2,:],Geom.histogram2d)
