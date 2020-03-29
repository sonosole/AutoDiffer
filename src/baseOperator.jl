# -- 变量节点的基本操作 --

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


function update(var::Variable, lr)
    # 更新单个 Variable
    var.value .-= lr .* var.delta
end


function update(vars::Array{Any,1}, lr)
    # 更新 Variable 数组
    for var in vars
        update(var, lr)
    end
end


function zerograds(parameters)
    for v in parameters
        v.delta .= 0.0
    end
end


# -----------------------------------------------
# 常用数学操作 点乘、点加、矩阵乘、数乘、数加 .......
# -----------------------------------------------
import Base.+
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


function Base.:+(var::Variable, constant)
    # 矩阵、向量与常数按元素相加
    out = Variable(var.value .+ constant, var.trainable)
    if var.trainable
        function addBackward()
            var.delta += out.delta
        end
        push!(graph.backward, addBackward)
    end
    return out
end


function Base.:+(constant, var::Variable)
    # 矩阵、向量与常数按元素相加
    out = Variable(var.value .+ constant, var.trainable)
    if var.trainable
        function addBackward()
            var.delta += out.delta
        end
        push!(graph.backward, addBackward)
    end
    return out
end


function Base.:*(var::Variable, constant)
    # 矩阵、列向量与常数按元素相乘
    out = Variable(var.value .* constant, var.trainable)
    if var.trainable
        function scalarMulMatBackward()
            var.delta += out.delta .* constant
        end
        push!(graph.backward, scalarMulMatBackward)
    end
    return out
end


function Base.:*(constant, var::Variable)
    # 矩阵、列向量与常数按元素相乘
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
            var.delta += n .* out.value ./ (var.value .+ 1e-200) .* out.delta
        end
        push!(graph.backward, powerBackward)
    end
    return out
end


function Base.:vcat(var1::Variable, var2::Variable)
    row1, col = size(var1.value)
    row2, col = size(var2.value)
    trainable = (var1.trainable || var2.trainable)
    out = Variable(vcat(var1.value, var2.value), trainable)

    if trainable
        function vcatBackward()
            idx1 = 1:row1
            idx2 = (row1+1):(row1+row2)
            var1.delta += out.delta[idx1,:]
            var2.delta += out.delta[idx2,:]
        end
        push!(graph.backward, vcatBackward)
    end
    return out
end


function dotAdd(var1::Variable, var2::Variable)
    # 相同形状的矩阵或者向量按对应位置的元素相加，即点加操作
    @assert(size(var1.value) == size(var2.value))
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


function Base.:+(var1::Variable, var2::Variable)
    # 相同形状的矩阵或者向量按对应位置的元素相加，即点加操作
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


function dotMul(var1::Variable, var2::Variable)
    # 相同形状的矩阵或者向量按对应位置的元素相乘,即点乘操作
    @assert(size(var1.value) == size(var2.value))
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
    # 矩阵相乘 A[i,j] = sum(B[i,k]*C[k,j],k=...)
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
    @assert(size(var1.value,1)==size(var2.value,1) && size(var2.value,2)==1)
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
    @assert(size(var1.value,1)==size(var2.value,1) && size(var2.value,2)==1)
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
# 常用激活函数 relu leakyrelu sigmoid TANH SIN COS
# -----------------------------------------------
function relu(var::Variable)
    out = Variable(max.(0.0, var.value), var.trainable)
    if var.trainable
        function reluBackward()
            var.delta += max.(0.0, var.value) ./ var.value .* out.delta
        end
        push!(graph.backward, reluBackward)
    end
    return out
end


function relu(x::Array)
    return max.(0.0, x)
end


function leakyrelu(var::Variable)
    out = Variable(max.(0.1 .* var.value, var.value), var.trainable)
    if var.trainable
        function leakyreluBackward()
            var.delta = max.(0.1 .* var.value, var.value) ./ var.value .* out.delta
        end
        push!(graph.backward, leakyreluBackward)
    end
    return out
end


function leakyrelu(x::Array)
    return max.(0.1 .* x, x)
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


function sigmoid(x::Array)
    return 1.0 ./ (1.0 .+ exp.(-x))
end


function swish(var::Variable)
    return dotMul(sigmoid(var), var)
end


function swish(x::Array)
    return x ./ (1.0 .+ exp.(-x))
end


function softmax(var::Variable)
    row, col = size(var.value)
    out = Variable(zeros(row, col), var.trainable)

    Xmax = maximum(var.value, dims=1)
    out.value = exp.(var.value .- Xmax)
    out.value ./= sum(out.value, dims=1)

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


function softmax(x::Array)
    xmax = maximum(x, dims=1)
    prob = exp.(x .- xmax)
    psum = sum(prob, dims=1)
    return (prob ./ psum)
end


function crossEntropy(var::Variable, label::Variable)
    @assert( size(var.value) == size(label.value) )
    trainable = (var.trainable || label.trainable)
    out = Variable(- label.value .* log.(var.value .+ 1e-200), trainable)
    if trainable
        function crossEntropyBackward()
            var.delta += - label.value ./ (var.value .+ 1e-200) .* out.delta
        end
        push!(graph.backward, crossEntropyBackward)
    end
    return out
end


function binaryCrossEntropy(var::Variable, label::Variable)
    @assert( size(var.value) == size(label.value) )
    trainable = (var.trainable || label.trainable)
    tmp1 = - label.value .* log.(var.value .+ 1e-200)
    tmp2 = - (1.0 .- label.value) .* log.(1.0 .- var.value .+ 1e-200)
    out  = Variable(tmp1 + tmp2, trainable)
    if trainable
        function binaryCrossEntropyBackward()
            temp1 = (1.0 .- label.value) ./ (1.0 .- var.value .+ 1e-200)
            temp2 = label.value ./ (var.value .+ 1e-200)
            var.delta += out.delta .* (temp1 - temp2)
        end
        push!(graph.backward, binaryCrossEntropyBackward)
    end
    return out
end


function mse(var::Variable, label::Variable)
    @assert( size(var.value) == size(label.value) )
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


function softplus(x::Array)
    return log.( 1.0 .+ exp.(x) )
end


function Base.:exp(var::Variable)
    out = Variable(exp.(var.value), var.trainable)
    if var.trainable
        function expBackward()
            var.delta += exp.(var.value) .* out.delta
        end
        push!(graph.backward, expBackward)
    end
    return out
end


function Base.:exp(x::Array)
    return exp.(x)
end


function Base.:log(var::Variable)
    out = Variable(log.(var.value), var.trainable)
    if var.trainable
        function logBackward()
            var.delta += out.delta ./ (var.value .+ 1e-200)
        end
        push!(graph.backward, logBackward)
    end
    return out
end


function Base.:log(x::Array)
    return log.(x)
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


function Base.:abs(x)
    return abs.(x)
end


function Base.:reshape(var::Variable, newsize)
    out = Variable( reshape(var.value, newsize), var.trainable )
    if var.trainable
        function reshapeBackward()
            var.delta .= reshape(out.delta, size(var.value))
        end
        push!(graph.backward, reshapeBackward)
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


function Base.:exp2(x::Array)
    return exp2.(x)
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


function Base.:exp10(x::Array)
    return exp10.(x)
end


function Base.:log2(var::Variable)
    # LOG10 represents y = log2(x)
    out = Variable(log10.(var.value), var.trainable)
    if var.trainable
        function log2Backward()
            var.delta += out.delta ./ (log(2.0) .* (var.value .+ 1e-200))
        end
        push!(graph.backward, log2Backward)
    end
    return out
end


function Base.:log2(x::Array)
    return log2.(x)
end


function Base.:log10(var::Variable)
    # LOG10 represents y = log10(x)
    out = Variable(log10.(var.value), var.trainable)
    if var.trainable
        function log10Backward()
            var.delta += out.delta ./ (log(10.0) .* (var.value .+ 1e-200))
        end
        push!(graph.backward, log10Backward)
    end
    return out
end


function Base.:log10(x::Array)
    return log10.(x)
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


function Base.:sec(x::Array)
    return sec.(x)
end


function Base.:sqrt(var::Variable)
    # SQRT represents y = sqrt(x)
    out = Variable(sqrt.(var.value), var.trainable)
    if var.trainable
        function sqrtBackward()
            var.delta += out.delta ./ (2.0 .* (var.value .+ 1e-200))
        end
        push!(graph.backward, sqrtBackward)
    end
    return out
end


function Base.:sqrt(x::Array)
    return sqrt.(x)
end


# -- tan serials --
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



function Base.:tan(x::Array)
    return tan.(x)
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


function Base.:tand(x::Array)
    return tand.(x)
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


function Base.:tanh(x::Array)
    return tanh.(x)
end


# # -- sin serials --
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


function Base.:sin(x::Array)
    return sin.(x)
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


function Base.:sinc(x::Array)
    return sinc.(x)
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


function Base.:sind(x::Array)
    return sind.(x)
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


function Base.:sinpi(x::Array)
    return sinpi.(x)
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


function Base.:cos(::Array)
    return cos.(x)
end


function Base.:inv(var::Variable)
    out = Variable(inv.(var.value), var.trainable)
    if var.trainable
        function invBackward()
            var.delta += - out.delta ./ (var.value.^2 .+ 1e-200)
        end
        push!(graph.backward, invBackward)
    end
    return out
end


function Base.:inv(x::Array)
    return inv.(x)
end


# -- some Aggregate operators --
function maxAggregate(var::Variable)
    out = Variable(maximum(var.value, dims=2), var.trainable)
    mask = (var.value .== out.value)
    if var.trainable
        function maxAggregateBackward()
            var.delta += out.delta .* mask
        end
        push!(graph.backward, maxAggregateBackward)
    end
    return out
end


# # function maxAggregate(var::Array{Variable,1})
# #     T = length(var)             # timeSteps
# #     N = size(var[1].value, 2)   # batchSize
# #     out = Variable( var[1].value )
# #
# #     for i = 1:N
# #         for t = 1:T
# #             out.value[:,i] .= maximum([out.value[:,i] var[t].value[:,i]], dims=2)
# #         end
# #     end
# #
# #     if graph.backProp
# #         function backprop()
# #             for t = 1:T
# #                 var[t].delta += .... out.delta
# #             end
# #         end
# #         out.backward = Backward
# #         push!(graph.operator, maxAggregate)
# #         for t = 1:T
# #             out.value[:,i] .= maximum([out.value[:,i] var[t].value[:,i]], dims=2)
# #             push!(var[t].children, out)
# #             push!(out.parents, var[t])
# #         end
# #     return out
# # end
#
#
function meanAggregate(var::Variable)
    fac = 1.0 / size(var.value, 2)
    out = Variable(sum(var.value, dims=2) .* fac, var.trainable)
    if var.trainable
        function meanAggregateBackward()
            var.delta += fac .* out.delta
        end
        push!(graph.backward, meanAggregateBackward)
    end
    return out
end


function linearAggregate(var::Variable)
    vsum1 = sum(var.value .* var.value, dims=2)
    vsum2 = sum(var.value, dims=2) .+ 1e-200
    out = Variable(vsum1 ./ vsum2, var.trainable)

    if var.trainable
        function linearAggregateBackward()
            var.delta .+= (2 .* var.value .- out.value) ./ vsum2 .* out.delta
        end
        push!(graph.backward, linearAggregateBackward)
    end
    return out
end


function expAggregate(var::Variable)
    temp  = exp.(var.value)
    vsum1 = sum(temp .* var.value, dims=2)
    vsum2 = sum(temp, dims=2) .+ 1e-200
    out = Variable(vsum1 ./ vsum2, var.trainable)

    if var.trainable
        function expAggregateBackward()
            var.delta .+= temp ./ vsum2 .* (1.0 .+ var.value .- out.value) .* out.delta
        end
        push!(graph.backward, expAggregateBackward)
    end
    return out
end
