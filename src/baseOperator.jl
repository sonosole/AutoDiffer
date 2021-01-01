# -- 变量节点的基本操作 --
# -- basic operators about Variable --
import Base.getindex
import Base.setindex!


Base.getindex(var::Variable, k...) = var.value[k...]
Base.setindex!(var::Variable, v, i...) = (var.value[k...] = v)


function showvar(var::Variable)
    print(valueof(var),"\n")
    print(gradof(var),"\n")
    println("trainable:",var.trainable,"\n")
end


function gradof(var::Variable)
    return var.delta
end


function valueof(var::Variable)
    return var.value
end


function backward()
    for i = length(graph.backward):-1:1
        graph.backward[i]()
    end
    clear()
end


function clear()
    graph.backward = []
end


function clip(x, clipval)
    x = (abs(x) > clipval) ? clipval * sign(x) : x
end


function update(var::Variable, lr)
    # update single Variable
    @. var.value -= lr * var.delta
end


function update(vars::Vector{Variable}, lr)
    # update multi Variables
    for var in vars
        update(var, lr)
    end
end


function zerograds(parameters)
    for v in parameters
        v.delta .= 0.0
    end
end


function eyes(n::Int)
    eye = zeros(n,n)
    for i = 1:n
        eye[i,i] = 1.0
    end
    return eye
end


function uniform(shape::Tuple;from=0.0,to=1.0)
    if from==0.0 && to==1.0
        return rand(typeof(from),shape)
    else
        return rand(typeof(from),shape) .* (to - from) .+ from
    end
end




# ------------------------------------------------
# 常用数学操作 点乘、点加、矩阵乘、数乘、数加 ........
# basic math operators like dot mul,dot add,etc ..
# ------------------------------------------------
import Base.+
import Base.-
import Base.*
import Base.^

import Base.cos
import Base.log
import Base.abs
import Base.reshape
import Base.vcat

import Base.exp
import Base.exp2
import Base.exp10

import Base.log2
import Base.log10

import Base.sec
import Base.sqrt

import Base.tan
import Base.tanh
import Base.tanh

import Base.sin
import Base.sinc
import Base.sind
import Base.sinpi
import Base.inv
import Base.size
import Base.ndims
import Base.length


function Base.:size(var::Variable)
    return size(var.value)
end


function Base.:size(var::Variable, dim::Int)
    return size(var.value, dim)
end


function Base.:length(var::Variable)
    return length(var.value)
end


function Base.:ndims(var::Variable)
    return ndims(var.value)
end


function Base.:+(var::Variable, constant)
    # a matrix add a constant element by element
    out = Variable(var.value .+ constant, var.trainable)
    if var.trainable
        function matAddScalarBackward()
            var.delta += out.delta
        end
        push!(graph.backward, matAddScalarBackward)
    end
    return out
end


function Base.:+(constant, var::Variable)
    # a matrix add a constant element by element
    out = Variable(constant .+ var.value, var.trainable)
    if var.trainable
        function scalarAddMatBackward()
            var.delta += out.delta
        end
        push!(graph.backward, scalarAddMatBackward)
    end
    return out
end


function Base.:-(var::Variable, constant)
    # a matrix minus a constant element by element
    out = Variable(var.value .- constant, var.trainable)
    if var.trainable
        function matMinusScalarBackward()
            var.delta += out.delta
        end
        push!(graph.backward, matMinusScalarBackward)
    end
    return out
end


function Base.:-(constant, var::Variable)
    # a matrix minus a constant element by element
    out = Variable(constant .- var.value, var.trainable)
    if var.trainable
        function scalarMinusMatBackward()
            var.delta -= out.delta
        end
        push!(graph.backward, scalarMinusMatBackward)
    end
    return out
end


function Base.:*(var::Variable, constant)
    # a matrix multiplies a constant element by element
    out = Variable(var.value .* constant, var.trainable)
    if var.trainable
        function matMulScalarBackward()
            var.delta += out.delta .* constant
        end
        push!(graph.backward, matMulScalarBackward)
    end
    return out
end


function Base.:*(constant, var::Variable)
    # a matrix multiplies a constant element by element
    out = Variable(var.value .* constant, var.trainable)
    if var.trainable
        function scalarMulMatBackward()
            var.delta += out.delta .* constant
        end
        push!(graph.backward, scalarMulMatBackward)
    end
    return out
end


function Base.:^(var::Variable, n::Int)
    # 矩阵、列向量与常数按元素做幂指数运算
    out = Variable(var.value .^ n, var.trainable)
    if var.trainable
        function powerBackward()
            var.delta += n .* out.value ./ (var.value .+ 1e-38) .* out.delta
        end
        push!(graph.backward, powerBackward)
    end
    return out
end


function indexbounds(sizeArray)
	# assert sizeArray has no 0 element
    acc = 0
    num = length(sizeArray)
    s = ones(Int,num,1)
    e = ones(Int,num,1)
    for i = 1:num
        s[i] += acc
        e[i] = s[i] + sizeArray[i] - 1
        acc += sizeArray[i]
    end
    return (s,e)
end


function Base.:vcat(var1::Variable, var2::Variable)
    row1 = size(var1,1)
    row2 = size(var2,1)
    trainable = (var1.trainable || var2.trainable)
    out = Variable(vcat(var1.value, var2.value), trainable)

    if trainable
        function vcatBackward()
            idx1 = 1:row1
            idx2 = (row1+1):(row1+row2)
            dims  = ndims(out)
            shape =  size(out)
            var1.delta += out.delta[idx1,CartesianIndices(shape[2:dims])]
            var2.delta += out.delta[idx2,CartesianIndices(shape[2:dims])]
        end
        push!(graph.backward, vcatBackward)
    end
    return out
end





function vcats(vars::Vector{Variable})
    outdims,batchsize = size(vars[1])

    timeSteps = length(vars)
    trainable = vars[1].trainable

    z = zeros(0,batchsize)
    for t = 1:timeSteps
        z = vcat(z,vars[t].value)
    end
    z = reshape(z,(outdims,timeSteps,batchsize))
    out = Variable(z, trainable)

    if trainable
        function vcatsBackward()
            for t = 1:timeSteps
                vars[t].delta += out.delta[:,t,:]
            end
        end
        push!(graph.backward, vcatsBackward)
    end
    return out
end


function Base.:+(var1::Variable, var2::Variable)
    # a matrix add a matrix element by element
    @assert (size(var1) == size(var2)) "2 inputs shall be the same size"
    trainable = (var1.trainable || var2.trainable)
    out = Variable(var1.value + var2.value, trainable)
    if trainable
        function add2varBackward()
            var1.delta += out.delta
            var2.delta += out.delta
        end
        push!(graph.backward, add2varBackward)
    end
    return out
end


function Base.:-(var1::Variable, var2::Variable)
    # a matrix minus a matrix element by element
    @assert (size(var1) == size(var2)) "2 inputs shall be the same size"
    trainable = (var1.trainable || var2.trainable)
    out = Variable(var1.value - var2.value, trainable)
    if trainable
        function minus2varBackward()
            var1.delta += out.delta
            var2.delta -= out.delta
        end
        push!(graph.backward, minus2varBackward)
    end
    return out
end


function dotAdd(var1::Variable, var2::Variable)
    # a matrix add a matrix element by element
    trainable = (var1.trainable || var2.trainable)
    out = Variable(var1.value .+ var2.value, trainable)
    if trainable
        function dotAddBackward()
            var1.delta += out.delta
            var2.delta += out.delta
        end
        push!(graph.backward, dotAddBackward)
    end
    return out
end


function dotMul(var1::Variable, var2::Variable)
    # a matrix multiplies a matrix element by element
    @assert (size(var1) == size(var2)) "2 inputs shall be the same size"
    trainable = (var1.trainable || var2.trainable)
    out = Variable(var1.value .* var2.value, trainable)
    if trainable
        function dotMulBackward()
            var1.delta += out.delta .* var2.value
            var2.delta += out.delta .* var1.value
        end
        push!(graph.backward, dotMulBackward)
    end
    return out
end


function Base.:*(var1::Variable, var2::Variable)
    # matrix var1 multiplies matrix var2
    # 矩阵相乘 C[i,j] = sum(A[i,k]*B[k,j],k=...)
    # var1 -- 权重矩阵
    # var2 -- 一个 batch 的多个输入列向量组成的矩阵
    # out  -- 一个 batch 的多个输出列向量组成的矩阵
    trainable = (var1.trainable || var2.trainable)
    out = Variable(var1.value * var2.value, trainable)
    if trainable
        function matMulBackward()
            var1.delta += out.delta * var2.value'
            var2.delta += var1.value' * out.delta
        end
        push!(graph.backward, matMulBackward)
    end
    return out
end


function matAddVec(var1::Variable, var2::Variable)
    # var1 -- 充当和节点，非网络需要学习的参数
    # var2 -- 偏置列向量，是网络需要学习的参数
    @assert (size(var1,1)==size(var2,1) && size(var2,2)==1)
    trainable = (var1.trainable || var2.trainable)
    out = Variable(var1.value .+ var2.value, trainable)
    if trainable
        function matAddVecBackward()
            var1.delta += out.delta
            var2.delta += sum(out.delta, dims=2)
        end
        push!(graph.backward, matAddVecBackward)
    end
    return out
end


function matMulVec(var1::Variable, var2::Variable)
    # var1 -- 一般充当激活节点，非网络需要学习的参数
    # var2 -- 列向量，循环权重，是网络需要学习的参数
    @assert (size(var1,1)==size(var2,1) && size(var2,2)==1)
    trainable = (var1.trainable || var2.trainable)
    out = Variable(var1.value .* var2.value, trainable)
    if trainable
        function matMulVecBackward()
            var1.delta += out.delta .* var2.value
            var2.delta += sum(out.delta .* var1.value, dims=2)
        end
        push!(graph.backward, matMulVecBackward)
    end
    return out
end


# -----------------------------------------------
#      Non-linear activation functions relu
# -----------------------------------------------


function relu!(var::Variable)
    @. var.value = max(0.0, var.value)
    o2i = var.value .> 0.0
    out = Variable(var.value, var.trainable)
    if var.trainable
        function reluBackward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, reluBackward)
    end
    return out
end


function relu(var::Variable)
    o2i = var.value .> 0.0
    out = Variable(max.(0.0, var.value), var.trainable)
    if var.trainable
        function reluBackward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, reluBackward)
    end
    return out
end


function relu!(x::Array)
    @. x = max(0.0, x)
end


function relu(x::Array)
    return max.(0.0, x)
end


function relu1!(var::Variable)
    @. var.value = min(1.0, max(0.0, var.value))
    o2i = 0.0 .< var.value .< 1.0
    out = Variable(var.value, var.trainable)
    if var.trainable
        function relu1Backward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end


function relu1(var::Variable)
    o2i = 0.0 .< var.value .< 1.0
    out = Variable(min.(1.0, max.(0.0, var.value)), var.trainable)
    if var.trainable
        function relu1Backward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end


function relu1!(x::Array)
    @. x = min(1.0, max(0.0, x))
end


function relu1(x::Array)
    return min.(1.0, max.(0.0, x))
end


function relu6!(var::Variable)
    @. var.value = min(6.0, max(0.0, var.value))
    o2i = 0.0 .< var.value .< 6.0
    out = Variable(var.value, var.trainable)
    if var.trainable
        function relu1Backward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end


function relu6(var::Variable)
    o2i = 0.0 .< var.value .< 6.0
    out = Variable(min.(6.0, max.(0.0, var.value)), var.trainable)
    if var.trainable
        function relu1Backward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, relu1Backward)
    end
    return out
end


function relu6!(x::Array)
    @. x = min(6.0, max(0.0, x))
end


function relu6(x::Array)
    return min.(6.0, max.(0.0, x))
end


function line!(var::Variable)
    o2i = -1.0 .< var.value .< 1.0
    @. var.value = o2i * var.value
    out = Variable(var.value, var.trainable)
    if var.trainable
        function lineBackward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, lineBackward)
    end
    return out
end


function line(var::Variable)
    o2i = -1.0 .< var.value .< 1.0
    out = Variable(o2i .* var.value, var.trainable)
    if var.trainable
        function lineBackward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, lineBackward)
    end
    return out
end


function line!(x::Array)
    @. x = (-1.0 < x < 1.0) * x
end


function line(x::Array)
    return (-1.0 .< x .< 1.0) .* x
end


function hardtanh!(var::Variable)
    @. var.value = min(1.0, max(-1.0, var.value))
    o2i = abs(var.value) .< 1.0
    out = Variable(var.value, var.trainable)
    if var.trainable
        function hardtanhBackward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, hardtanhBackward)
    end
    return out
end


function hardtanh(var::Variable)
    o2i = abs(var.value) .< 1.0
    out = Variable(min.(1.0, max.(-1.0, var.value)), var.trainable)
    if var.trainable
        function hardtanhBackward()
            var.delta += out.delta .* o2i
        end
        push!(graph.backward, hardtanhBackward)
    end
    return out
end


function hardtanh!(x::Array)
    @. x = min(1.0, max(-1.0, x))
end


function hardtanh(x::Array)
    return min.(1.0, max.(-1.0, x))
end


function leakyrelu!(var::Variable)
    tempv = var.value .* 0.1
    mask1 = var.value .> tempv
    mask2 = .!mask1
    @. var.value = max(tempv, var.value)
    out  = Variable(var.value, var.trainable)
    if var.trainable
        function leakyreluBackward()
            var.delta = (mask1 + 0.1 .* mask2) .* out.delta
        end
        push!(graph.backward, leakyreluBackward)
    end
    return out
end


function leakyrelu(var::Variable)
    tempv = var.value .* 0.1
    mask1 = var.value .> tempv
    mask2 = .!mask1
    out  = Variable(max.(tempv, var.value), var.trainable)
    if var.trainable
        function leakyreluBackward()
            var.delta = (mask1 + 0.1 .* mask2) .* out.delta
        end
        push!(graph.backward, leakyreluBackward)
    end
    return out
end


function leakyrelu!(x::Array)
    @. x = max(0.1 * x, x)
end


function leakyrelu(x::Array)
    return max.(0.1 * x, x)
end


function sigmoid!(var::Variable)
    @. var.value = 1.0 / (1.0 + exp(-var.value))
    out = Variable(var.value , var.trainable)
    if var.trainable
        function sigmoidBackward()
            var.delta += out.value .* (1.0 .- out.value) .* out.delta
        end
        push!(graph.backward, sigmoidBackward)
    end
    return out
end


function sigmoid(var::Variable)
    out = Variable(1.0 ./ (1.0 .+ exp.(-var.value)) , var.trainable)
    if var.trainable
        function sigmoidBackward()
            var.delta += out.value .* (1.0 .- out.value) .* out.delta
        end
        push!(graph.backward, sigmoidBackward)
    end
    return out
end


function sigmoid!(x::Array)
    @. x = 1.0 / (1.0 + exp(-x))
end


function sigmoid(x::Array)
    return 1.0 ./ (1.0 .+ exp.(-x))
end


function swish!(var::Variable)
    return dotMul(sigmoid!(var), var)
end


function swish(var::Variable)
    return dotMul(sigmoid(var), var)
end


function swish!(x::Array)
    @. x = x / (1.0 + exp(-x))
end


function swish(x::Array)
    return  x ./ (1.0 .+ exp.(-x))
end


function softmax(var::Variable; dims=1)
    row, col = size(var)
    out = Variable(zeros(row, col), var.trainable)

    Xmax = maximum(var.value, dims=dims)
    out.value = exp.(var.value .- Xmax)
    out.value ./= sum(out.value, dims=dims)

    if var.trainable
        function softmaxBackward()
            for j = 1:col
                for i = 1:row
                    for k = 1:row
                        d = (k==i ? 1.0 : 0.0)
                        var.delta[i,j] += out.value[k,j] * (d - out.value[i,j]) * out.delta[k,j]
                    end
                end
            end
        end
        push!(graph.backward, softmaxBackward)
    end
    return out
end


function softmax(x::Array; dims=1)
    xmax = maximum(x, dims=dims)
    prob = exp.(x .- xmax)
    psum = sum(prob, dims=dims)
    return (prob ./ psum)
end


function crossEntropy(var::Variable, label::Variable)
    @assert ( size(var) == size(label) )
    trainable = (var.trainable || label.trainable)
    out = Variable(- label.value .* log.(var.value .+ 1e-38), trainable)
    if trainable
        function crossEntropyBackward()
            var.delta += - label.value ./ (var.value .+ 1e-38) .* out.delta
        end
        push!(graph.backward, crossEntropyBackward)
    end
    return out
end


function binaryCrossEntropy(var::Variable, label::Variable)
    @assert ( size(var) == size(label) )
    trainable = (var.trainable || label.trainable)
    tmp1 = - label.value .* log.(var.value .+ 1e-38)
    tmp2 = - (1.0 .- label.value) .* log.(1.0 .- var.value .+ 1e-38)
    out  = Variable(tmp1 + tmp2, trainable)
    if trainable
        function binaryCrossEntropyBackward()
            temp1 = (1.0 .- label.value) ./ (1.0 .- var.value .+ 1e-38)
            temp2 = label.value ./ (var.value .+ 1e-38)
            var.delta += out.delta .* (temp1 - temp2)
        end
        push!(graph.backward, binaryCrossEntropyBackward)
    end
    return out
end


function mse(var::Variable, label::Variable)
    @assert ( size(var) == size(label) )
    trainable = (var.trainable || label.trainable)
    out = Variable((var.value - label.value).^2, trainable)
    if trainable
        function mseBackward()
            var.delta += 2.0 .* (var.value - label.value) .* out.delta
        end
        push!(graph.backward, mseBackward)
    end
    return out
end


function cost(var::Variable)
    out = Variable(sum(var.value), var.trainable)
    out.delta = 1.0
    if var.trainable
        function costBackward()
            var.delta += var.delta .+ 1.0
        end
        push!(graph.backward, costBackward)
    end
    return out
end


function mseLoss(var::Variable, label::Variable)
    return cost( mse(var, label) )
end


function binaryCrossEntropyLoss(var::Variable, label::Variable)
    return cost( binaryCrossEntropy(var, label) )
end


function crossEntropyLoss(var::Variable, label::Variable)
    return cost( crossEntropy(var, label) )
end


# -----------------
# 不常用激活函数....
# -----------------
function softplus!(var::Variable)
    out = Variable(log.( 1.0 .+ exp.(var.value) ), var.trainable)
    if var.trainable
        function softplusBackward()
            var.delta += out.delta ./ (1.0 .+ exp.(-var.value))
        end
        push!(graph.backward, softplusBackward)
    end
    return out
end


function softplus(var::Variable)
    out = Variable(log.( 1.0 .+ exp.(var.value) ), var.trainable)
    if var.trainable
        function softplusBackward()
            var.delta += out.delta ./ (1.0 .+ exp.(-var.value))
        end
        push!(graph.backward, softplusBackward)
    end
    return out
end


function softplus!(x::Array)
    @. x = log(1.0 + exp(x))
end


function softplus(x::Array)
    return log.( 1.0 .+ exp.(x) )
end


function exp!(var::Variable)
    @. var.value = exp(var.value)
    out = Variable(var.value, var.trainable)
    if var.trainable
        function expBackward()
            var.delta += out.value .* out.delta
        end
        push!(graph.backward, expBackward)
    end
    return out
end


function Base.:exp(var::Variable)
    out = Variable(exp.(var.value), var.trainable)
    if var.trainable
        function expBackward()
            var.delta += out.value .* out.delta
        end
        push!(graph.backward, expBackward)
    end
    return out
end


function exp!(x::Array)
    @. x = exp(x)
end


function Base.:exp(x::Array)
    return exp.(x)
end


function log!(var::Variable)
    out = Variable(log.(var.value), var.trainable)
    if var.trainable
        function logBackward()
            var.delta += out.delta ./ var.value
        end
        push!(graph.backward, logBackward)
    end
    return out
end


function Base.:log(var::Variable)
    out = Variable(log.(var.value), var.trainable)
    if var.trainable
        function logBackward()
            var.delta += out.delta ./ var.value
        end
        push!(graph.backward, logBackward)
    end
    return out
end


function log!(x::Array)
    @. x = log(x)
end


function Base.:log(x::Array)
    return log.(x)
end


function abs!(var::Variable)
    out = Variable(abs.(var.value), var.trainable)
    if var.trainable
        function absBackward()
            var.delta += sign.(var.value) .* out.delta
        end
        push!(graph.backward, absBackward)
    end
    return out
end


function Base.:abs(var::Variable)
    out = Variable(abs.(var.value), var.trainable)
    if var.trainable
        function absBackward()
            var.delta += sign.(var.value) .* out.delta
        end
        push!(graph.backward, absBackward)
    end
    return out
end


function abs!(x)
    @. x = abs(x)
end


function Base.:abs(x)
    return abs.(x)
end


function Base.:reshape(var::Variable, newsize)
    out = Variable( reshape(var.value, newsize), var.trainable )
    if var.trainable
        function reshapeBackward()
            var.delta += reshape(out.delta, size(var.value))
        end
        push!(graph.backward, reshapeBackward)
    end
    return out
end


function exp2!(var::Variable)
    # EXP2 represents y = 2^x
    @. var.value = exp2(var.value)
    out = Variable(var.value, var.trainable)
    if var.trainable
        function exp2Backward()
            var.delta += log(2.0) .* out.value .* out.delta
        end
        push!(graph.backward, exp2Backward)
    end
    return out
end


function Base.:exp2(var::Variable)
    # EXP2 represents y = 2^x
    out = Variable(exp2.(var.value), var.trainable)
    if var.trainable
        function exp2Backward()
            var.delta += log(2.0) .* out.value .* out.delta
        end
        push!(graph.backward, exp2Backward)
    end
    return out
end


function exp2!(x::Array)
    @. x = exp2(x)
end


function Base.:exp2(x::Array)
    return exp2.(x)
end


function exp10!(var::Variable)
    # EXP10 represents y = 10^x
    @. var.value = exp10(var.value)
    out = Variable(var.value, var.trainable)
    if var.trainable
        function exp10Backward()
            var.delta += log(10.0) .* out.value .* out.delta
        end
        push!(graph.backward, exp10Backward)
    end
    return out
end


function Base.:exp10(var::Variable)
    # EXP10 represents y = 10^x
    out = Variable(exp10.(var.value), var.trainable)
    if var.trainable
        function exp10Backward()
            var.delta += log(10.0) .* out.value .* out.delta
        end
        push!(graph.backward, exp10Backward)
    end
    return out
end


function exp10!(x::Array)
    @. x = exp10(x)
end


function Base.:exp10(x::Array)
    return exp10.(x)
end


function log2!(var::Variable)
    # LOG10 represents y = log2(x)
    out = Variable(log10.(var.value), var.trainable)
    if var.trainable
        function log2Backward()
            var.delta += out.delta ./ (log(2.0) .* (var.value .+ 1e-38))
        end
        push!(graph.backward, log2Backward)
    end
    return out
end


function Base.:log2(var::Variable)
    # LOG10 represents y = log2(x)
    out = Variable(log10.(var.value), var.trainable)
    if var.trainable
        function log2Backward()
            var.delta += out.delta ./ (log(2.0) .* (var.value .+ 1e-38))
        end
        push!(graph.backward, log2Backward)
    end
    return out
end


function log2!(x::Array)
    @. x = log2(x)
end


function Base.:log2(x::Array)
    return log2.(x)
end


function log10!(var::Variable)
    # LOG10 represents y = log10(x)
    out = Variable(log10.(var.value), var.trainable)
    if var.trainable
        function log10Backward()
            var.delta += out.delta ./ (log(10.0) .* (var.value .+ 1e-38))
        end
        push!(graph.backward, log10Backward)
    end
    return out
end


function Base.:log10(var::Variable)
    # LOG10 represents y = log10(x)
    out = Variable(log10.(var.value), var.trainable)
    if var.trainable
        function log10Backward()
            var.delta += out.delta ./ (log(10.0) .* (var.value .+ 1e-38))
        end
        push!(graph.backward, log10Backward)
    end
    return out
end


function log10!(x::Array)
    @. x = log10(x)
end


function Base.:log10(x::Array)
    return log10.(x)
end


function sec!(var::Variable)
    # SEC represents y = sec(x)
    out = Variable(sec.(var.value), var.trainable)
    if var.trainable
        function secBackward()
            var.delta += out.delta .* out.value .* tan.(var.value)
        end
        push!(graph.backward, secBackward)
    end
    return out
end


function Base.:sec(var::Variable)
    # SEC represents y = sec(x)
    out = Variable(sec.(var.value), var.trainable)
    if var.trainable
        function secBackward()
            var.delta += out.delta .* out.value .* tan.(var.value)
        end
        push!(graph.backward, secBackward)
    end
    return out
end


function sec!(x::Array)
    @. x = sec(x)
end


function Base.:sec(x::Array)
    return sec.(x)
end


function sqrt!(var::Variable)
    # SQRT represents y = sqrt(x)
    @. var.value = sqrt(var.value)
    out = Variable(var.value, var.trainable)
    if var.trainable
        function sqrtBackward()
            var.delta += out.delta ./ (2.0 .* (out.value .+ 1e-38))
        end
        push!(graph.backward, sqrtBackward)
    end
    return out
end


function Base.:sqrt(var::Variable)
    # SQRT represents y = sqrt(x)
    out = Variable(sqrt.(var.value), var.trainable)
    if var.trainable
        function sqrtBackward()
            var.delta += out.delta ./ (2.0 .* (out.value .+ 1e-38))
        end
        push!(graph.backward, sqrtBackward)
    end
    return out
end


function sqrt!(x::Array)
    @. x = sqrt(x)
end


function Base.:sqrt(x::Array)
    return sqrt.(x)
end


# -- tan serials --
function tan!(var::Variable)
    @. var.value = tan(var.value)
    out = Variable(var.value, var.trainable)
    if var.trainable
        function tanBackward()
            var.delta += (1.0 .+ out.value.^2) .* out.delta
        end
        push!(graph.backward, tanBackward)
    end
    return out
end


function Base.:tan(var::Variable)
    out = Variable(tan.(var.value), var.trainable)
    if var.trainable
        function tanBackward()
            var.delta += (1.0 .+ out.value.^2) .* out.delta
        end
        push!(graph.backward, tanBackward)
    end
    return out
end


function tan!(x::Array)
    @. x = tan(x)
end


function Base.:tan(x::Array)
    return tan.(x)
end


function tand!(var::Variable)
    @. var.value = tand(var.value)
    out = Variable(var.value, var.trainable)
    if var.trainable
        function tandBackward()
            var.delta += pi/180 .* (1.0 .+ out.value.^2) .* out.delta
        end
        push!(graph.backward, tandBackward)
    end
    return out
end


function Base.:tand(var::Variable)
    out = Variable(tand.(var.value), var.trainable)
    if var.trainable
        function tandBackward()
            var.delta += pi/180 .* (1.0 .+ out.value.^2) .* out.delta
        end
        push!(graph.backward, tandBackward)
    end
    return out
end


function tand!(x::Array)
    @. x = tand(x)
end


function Base.:tand(x::Array)
    return tand.(x)
end


function tanh!(var::Variable)
    out = Variable(tanh.(var.value), var.trainable)
    if var.trainable
        function tanhBackward()
            var.delta += (1.0 .- out.value.^2) .* out.delta
        end
        push!(graph.backward, tanhBackward)
    end
    return out
end


function Base.:tanh(var::Variable)
    out = Variable(tanh.(var.value), var.trainable)
    if var.trainable
        function tanhBackward()
            var.delta += (1.0 .- out.value.^2) .* out.delta
        end
        push!(graph.backward, tanhBackward)
    end
    return out
end


function tanh!(x::Array)
    @. x = tanh(x)
end


function Base.:tanh(x::Array)
    return tanh.(x)
end


function tanhshrink!(var::Variable)
    return var - tanh(var)
end


function tanhshrink(var::Variable)
    return var - tanh(var)
end


function tanhshrink!(x::Array)
    @. x = x - tanh(x)
end


function tanhshrink(x::Array)
    return  x - tanh(x)
end


# # -- sin serials --
function sin!(var::Variable)
    out = Variable(sin.(var.value), var.trainable)
    if var.trainable
        function sinBackward()
            var.delta += cos.(var.value) .* out.delta
        end
        push!(graph.backward, sinBackward)
    end
    return out
end


function Base.:sin(var::Variable)
    out = Variable(sin.(var.value), var.trainable)
    if var.trainable
        function sinBackward()
            var.delta += cos.(var.value) .* out.delta
        end
        push!(graph.backward, sinBackward)
    end
    return out
end


function sin!(x::Array)
    @. x = sin(x)
end


function Base.:sin(x::Array)
    return sin.(x)
end


function sinc!(var::Variable)
    # sinc represents y = sin(pi*x)/(pi*x)
    out = Variable(sinc.(var.value), var.trainable)
    if var.trainable
        function sincBackward()
            var.delta += cosc.(var.value) .* out.delta
        end
        push!(graph.backward, sincBackward)
    end
    return out
end


function Base.:sinc(var::Variable)
    # sinc represents y = sin(pi*x)/(pi*x)
    out = Variable(sinc.(var.value), var.trainable)
    if var.trainable
        function sincBackward()
            var.delta += cosc.(var.value) .* out.delta
        end
        push!(graph.backward, sincBackward)
    end
    return out
end


function sinc!(x::Array)
    @. x = sinc(x)
end


function Base.:sinc(x::Array)
    return sinc.(x)
end


function sind!(var::Variable)
    out = Variable(sind.(var.value), var.trainable)
    if var.trainable
        function sindBackward()
            var.delta += pi/180 .* cosd.(var.value) .* out.delta
        end
        push!(graph.backward, sindBackward)
    end
    return out
end


function Base.:sind(var::Variable)
    out = Variable(sind.(var.value), var.trainable)
    if var.trainable
        function sindBackward()
            var.delta += pi/180 .* cosd.(var.value) .* out.delta
        end
        push!(graph.backward, sindBackward)
    end
    return out
end


function sind!(x::Array)
    @. x = sind(x)
end


function Base.:sind(x::Array)
    return sind.(x)
end


function sinpi!(var::Variable)
    out = Variable(sinpi.(var.value), var.trainable)
    if var.trainable
        function sinpiBackward()
            var.delta += pi .* cospi.(var.value) .* out.delta
        end
        push!(graph.backward, sinpiBackward)
    end
    return out
end


function Base.:sinpi(var::Variable)
    out = Variable(sinpi.(var.value), var.trainable)
    if var.trainable
        function sinpiBackward()
            var.delta += pi .* cospi.(var.value) .* out.delta
        end
        push!(graph.backward, sinpiBackward)
    end
    return out
end


function sinpi!(x::Array)
    @. x = sinpi(x)
end


function Base.:sinpi(x::Array)
    return sinpi.(x)
end


function linearsin!(var::Variable)
    return sin(var) + var
end


function linearsin(var::Variable)
    return sin(var) + var
end


function linearsin!(x::Array)
    @. x = sin(x) + x
end


function linearsin(x::Array)
    return sin(x) + x
end


function cos!(var::Variable)
    out = Variable(cos.(var.value), var.trainable)
    if var.trainable
        function cosBackward()
            var.delta += - sin.(var.value) .* out.delta
        end
        push!(graph.backward, cosBackward)
    end
    return out
end


function Base.:cos(var::Variable)
    out = Variable(cos.(var.value), var.trainable)
    if var.trainable
        function cosBackward()
            var.delta += - sin.(var.value) .* out.delta
        end
        push!(graph.backward, cosBackward)
    end
    return out
end


function cos!(x::Array)
    @. x = cos(x)
end


function Base.:cos(::Array)
    return cos.(x)
end


function inv!(var::Variable)
    @. var.value = inv(var.value)
    out = Variable(var.value, var.trainable)
    if var.trainable
        function invBackward()
            var.delta += - out.delta .* out.value.^2
        end
        push!(graph.backward, invBackward)
    end
    return out
end


function Base.:inv(var::Variable)
    out = Variable(inv.(var.value), var.trainable)
    if var.trainable
        function invBackward()
            var.delta += - out.delta .* out.value.^2
        end
        push!(graph.backward, invBackward)
    end
    return out
end


function inv!(x::Array)
    @. x = inv(x)
end


function Base.:inv(x::Array)
    return inv.(x)
end


# -- some Aggregate operators --
function maxpool(var::Variable)
    out = Variable(maximum(var.value, dims=2), var.trainable)
    mask = (var.value .== out.value)
    if var.trainable
        function maxpoolBackward()
            var.delta += out.delta .* mask
        end
        push!(graph.backward, maxpoolBackward)
    end
    return out
end


function maxpool(x::Array)
    return maximum(x, dims=2)
end


function meanpool(var::Variable)
    fac = 1.0 / size(var, 2)
    out = Variable(fac*sum(var.value, dims=2), var.trainable)
    if var.trainable
        function meanpoolBackward()
            var.delta .+= fac * out.delta
        end
        push!(graph.backward, meanpoolBackward)
    end
    return out
end


function meanpool(x::Array)
    return (1.0/size(x, 2)) * sum(x, dims=2)
end


function linearpool(var::Variable)
    vsum1 = sum(var.value .* var.value, dims=2)
    vsum2 = sum(var.value, dims=2) .+ 1e-38
    out   = Variable(vsum1 ./ vsum2, var.trainable)
    if var.trainable
        function linearpoolBackward()
            var.delta += (2 .* var.value .- out.value) ./ vsum2 .* out.delta
        end
        push!(graph.backward, linearpoolBackward)
    end
    return out
end


function linearpool(x::Array)
    vsum1 = sum(x .* x, dims=2)
    vsum2 = sum(x, dims=2) .+ 1e-38
    return vsum1 ./ vsum2
end


function exppool(var::Variable)
    temp  = exp.(var.value)
    vsum1 = sum(temp .* var.value, dims=2)
    vsum2 = sum(temp, dims=2) .+ 1e-38
    out   = Variable(vsum1 ./ vsum2, var.trainable)

    if var.trainable
        function exppoolBackward()
            var.delta += temp ./ vsum2 .* (1.0 .+ var.value .- out.value) .* out.delta
        end
        push!(graph.backward, exppoolBackward)
    end
    return out
end


function exppool(x::Array)
    temp  = exp.(x)
    vsum1 = sum(temp .* x, dims=2)
    vsum2 = sum(temp, dims=2) .+ 1e-38
    return vsum1 ./ vsum2
end
