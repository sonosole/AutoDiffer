include("../src/AutoDiffer.jl")

# 测试 relu
m = [ 1. -2.;
     -3.  4.;
      5. -6.]
x = Variable(m,true)
y = relu(x)
y.delta = [-0.1 0.2;
            0.3 0.4;
            0.5 0.6]

backward()
print(x.delta)
# [-0.1 0.0;
#   0.0 0.4;
#   0.5 0.0]

# 测试 leakyrelu
m = [ 1. -2.;
     -3.  4.;
      5. -6.]
x = Variable(m,true)
y = leakyrelu(x)
y.delta = [-0.1 0.2;
            0.3 0.4;
            0.5 0.6]
backward()
print(x.delta)


# 测试 softmax
m = [ 1. 2.;
      1. 2.;
      1. 2.;
      1. 2.;]
x = Variable(m,true)
y = x * 0.1
z = softmax(y)
z.delta =-[1.1 1.9;
           1.2 1.8;
           1.3 1.7;
           1.4 1.6]
backward()
print(x.delta)


# 测试简单的前馈网络的梯度
input = randn(256,4).*2
l     = zeros(4,4)
l[1,1] = 1.0
l[2,2] = 1.0
l[3,3] = 1.0
l[4,4] = 1.0
label  = l ./ sum(l,dims=1)

param    = Vector(undef,4)
param[1] = randn(128,256)./128
param[2] = zeros(128,1)
param[3] = randn(4,128)./10
param[4] = zeros(4,1)

l = Variable(label) # label
x = Variable(input) # input

w1 = Variable(param[1],true) # weight1
b1 = Variable(param[2],true) # bias1
w2 = Variable(param[3],true) # weight2
b2 = Variable(param[4],true) # bias2
x1 = relu(matAddVec(w1*x, b1))
x2 = softmax(matAddVec(w2*x1, b2))
loss = crossEntropy(x2^10, l)
LOSS1 = cost(loss)
backward()
backwardGrad = w1.delta[1,1]


delta = 1e-6
param[1][1,1] += delta

x1 = relu(matAddVec(w1*x, b1))
x2 = softmax(matAddVec(w2*x1, b2))
loss = crossEntropy(x2^10, l)
LOSS2 = cost(loss)
backward()

forwardGrad = ( LOSS2.value - LOSS1.value )/delta
println("\n---------------------")
println(" forwardGrad:",  forwardGrad)
println("backwardGrad:", backwardGrad)
println("---------------------")
