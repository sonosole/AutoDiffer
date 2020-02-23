# -- 节点与图的基本操作 --

function showvar(var::Variable)
    println("value:\n",var.value,"\n----------------------------------------")
    println("delta:\n",var.delta,"\n----------------------------------------")
    println("trainable:",var.trainable,"\n----------------------------------------")
    println(var.parents)
    println(var.children)
end


function gradof(var::Variable)
    return var.delta
end


function valueof(var::Variable)
    return var.value
end


function Backward(graph::Graph)
    for i = length(graph.backward):-1:1
        graph.backward[i]()
    end
end


function update(var::Variable, lr)
    if var.trainable
        var.value .= var.value .- lr*var.delta
    end
end


# -----------------------------------------------
# 常用数学操作 点乘、点加、矩阵乘、数乘、数加........
# -----------------------------------------------
function scalarAddMat(graph::Graph, var::Variable, constant::Float64)
    # 矩阵、向量与常数按元素相加
    @assert(var.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    out = Variable( var.value .+ constant )
    if graph.backProp
        function backprop()
            var.delta += out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, scalarAddMat)
        push!(var.children, out)
        push!(out.parents, var)
        push!(out.parents, constant)
    end
    return out
end


function scalarAddMat(graph::Graph, constant::Float64, var::Variable)
    # 矩阵、向量与常数按元素相加
    @assert(var.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    out = Variable( var.value .+ constant )
    if graph.backProp
        function backprop()
            var.delta += out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, scalarAddMat)
        push!(var.children, out)
        push!(out.parents, var)
        push!(out.parents, constant)
    end
    return out
end


function concat(graph::Graph, var1::Variable, var2::Float64)
    row1, col1 = size(var1.value)
    row2, col2 = size(var2.value)
    @assert(col1 == col2)
    out = Variable( vcat(var1.value, var2.value) )
    if graph.backProp
        function backprop()
            idx1 = 1:row1
            idx2 = (row1+1):(row1+row2)
            var1.delta += out.delta[idx1,:]
            var2.delta += out.delta[idx2,:]
        end
        push!(graph.backward, backprop)
        push!(graph.operator, concat)
        push!(var1.children, out)
        push!(var2.children, out)
        push!(out.parents, var1)
        push!(out.parents, var2)
    end
    return out
end


function scalarMulMat(graph::Graph, var::Variable, constant::Float64)
    # 矩阵、列向量与常数按元素相乘
    @assert(var.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    out = Variable( var.value .* constant )
    if graph.backProp
        function backprop()
            var.delta += out.delta .* constant
        end
        push!(graph.backward, backprop)
        push!(graph.operator, scalarMulMat)
        push!(var.children, out)
        push!(out.parents, var)
        push!(out.parents, constant)
    end
    return out
end


function scalarMulMat(graph::Graph, constant::Float64, var::Variable)
    # 矩阵、列向量与常数按元素相乘
    @assert(var.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    out = Variable( var.value .* constant )
    if graph.backProp
        function backprop()
            var.delta += out.delta .* constant
        end
        push!(graph.backward, backprop)
        push!(graph.operator, scalarMulMat)
        push!(var.children, out)
        push!(out.parents, var)
        push!(out.parents, constant)
    end
    return out
end


function dotMul(graph::Graph, var1::Variable, var2::Variable)
    # 相同形状的矩阵或者向量按对应位置的元素相乘,即点乘操作
    @assert( size(var1.value) == size(var2.value) )
    @assert(var1.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    @assert(var2.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    out = Variable( var1.value .* var2.value )
    if graph.backProp
        function backprop()
            var1.delta += out.delta .* var2.value
            var2.delta += out.delta .* var1.value
        end
        push!(graph.backward, backprop)
        push!(graph.operator, dotMul)
        push!(var1.children, out)
        push!(var2.children, out)
        push!(out.parents, var1)
        push!(out.parents, var2)
    end
    return out
end


function dotAdd(graph::Graph, var1::Variable, var2::Variable)
    # 相同形状的矩阵或者向量按对应位置的元素相加，即点加操作
    @assert( size(var1.value) == size(var2.value) )
    @assert(var1.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    @assert(var2.trainable == false) # 属于激活函数的输入或者输出，而非网络需要学习的参数
    out = Variable( var1.value .+ var2.value )
    if graph.backProp
        function backprop()
            var1.delta += out.delta
            var2.delta += out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, dotAdd)
        push!(var1.children, out)
        push!(var2.children, out)
        push!(out.parents, var1)
        push!(out.parents, var2)
    end
    return out
end


function matMul(graph::Graph, var1::Variable, var2::Variable)
    # 矩阵相乘 A[i,j] = sum(B[i,k]*C[k,j],k=...)
    # var1 -- 权重矩阵
    # var2 -- 一个batch的多个输入列向量组成的矩阵
    # out  -- 一个batch的多个输出列向量组成的矩阵
    out = Variable( var1.value * var2.value )
    if graph.backProp
        function backprop()
            var1.delta += out.delta * var2.value'
            var2.delta += var1.value' * out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, matMul)
        push!(var1.children, out)
        push!(var2.children, out)
        push!(out.parents, var1)
        push!(out.parents, var2)
    end
    return out
end


function matAddVec(graph::Graph, var1::Variable, var2::Variable)
    # var1 -- 充当和节点
    # var2 -- 偏置列向量
    @assert(var1.trainable == false) # 属于激活函数的输入，非网络需要学习的参数
    @assert(var2.trainable == true)  # 充当网络的偏置参数，是网络需要学习的参数
    @assert(size(var1.value,1)==size(var2.value,1) && size(var2.value,2)==1)
    out = Variable( var1.value .+ var2.value )
    if graph.backProp
        function backprop()
            var1.delta += out.delta                 # 充当不可训练变量，如和节点
            var2.delta += sum(out.delta, dims=2)    # 充当网络的偏置参数
        end
        push!(graph.backward, backprop)
        push!(graph.operator, matAddVec)
        push!(var1.children, out)
        push!(var2.children, out)
        push!(out.parents, var1)
        push!(out.parents, var2)
    end
    return out
end


function matMulVec(graph::Graph, var1::Variable, var2::Variable)
    # var1 -- 一般充当激活节点
    # var2 -- 列向量，循环权重
    @assert(var1.trainable == false) # 属于激活函数的输出，非网络需要学习的参数
    @assert(var2.trainable == true)  # 充当网络的偏置参数，是网络需要学习的参数
    @assert(size(var1.value,1)==size(var2.value,1) && size(var2.value,2)==1)
    out = Variable( var1.value .* var2.value )
    if graph.backProp
        function backprop()
            var1.delta += out.delta .* var2.value              # 充当不可训练变量，如激活节点
            var2.delta += sum(out.delta .* var1.value, dims=2) # 充当网络的偏置参数
        end
        push!(graph.backward, backprop)
        push!(graph.operator, matMulVec)
        push!(var1.children, out)
        push!(var2.children, out)
        push!(out.parents, var1)
        push!(out.parents, var2)
    end
    return out
end


# -----------------------------------------------
# 常用激活函数 relu leakyrelu sigmoid TANH SIN COS
# -----------------------------------------------
function relu(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( max.(0.0, var.value) )
    if graph.backProp
        function backprop()
            var.delta += max.(0.0, var.value) ./ var.value .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, relu)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function _relu_(x)
    y = max.(0.0, x)
end


function leakyrelu(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( max.(0.1 .* var.value, var.value) )
    if graph.backProp
        function backprop()
            var.delta = max.(0.1 .* var.value, var.value) ./ var.value .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, leakyrelu)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function _leakyrelu_(x)
    y = max.(0.1 .* x, x)
end


function sigmoid(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( 1.0 ./ (1.0 .+ exp.(-var.value)) )
    if graph.backProp
        function backprop()
            var.delta += out.value .* (1.0 .- out.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, sigmoid)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function _sigmoid_(x)
    y = 1.0 ./ (1.0 .+ exp.(-x))
end


function swish(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    return dotMul(graph, sigmoid(graph, var), var)
end


function _swish_(args)
    y = x ./ (1.0 .+ exp.(-x))
end








function COS(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    row, col = size(var.value)
    out = Variable( cos.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += - sin.(var.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, COS)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function _COS_(x)
    y = cos.(x)
end


function softmax(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    row, col = size(var.value)
    out = Variable(zeros(row, col))

    Xmax = maximum(var.value, dims=1);
    out.value = exp.(var.value .- Xmax);
    Xsum = sum(out.value, dims=1);
    out.value ./= Xsum;

    if graph.backProp
        function backprop()
            for j = 1:col
                for i = 1:row
                    for k = 1:row
                        d = (k==i ? 1.0 : 0.0)
                        var.delta[i,j] += out.delta[k,j] * out.value[k,j] * (d - out.value[i,j])
                    end
                end
            end
        end
        push!(graph.backward, backprop)
        push!(graph.operator, softmax)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function _softmax_(x)
    Xmax = maximum(x, dims=1);
    y = exp.(x .- Xmax);
    Xsum = sum(y, dims=1);
    y ./= Xsum;
end


function crossEntropy(graph::Graph, var::Variable, label::Variable)
    @assert(var.trainable == false)
    @assert(label.trainable == false)
    @assert( size(var.value) == size(label.value) )
    out = Variable( - label.value .* log.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += (- label.value ./ var.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, crossEntropy)
        push!(var.children, out)
        push!(label.children, out)
        push!(out.parents, var)
        push!(out.parents, label)
    end
    return out
end


function binaryCrossEntropy(graph::Graph, var::Variable, label::Variable)
    @assert(var.trainable == false)
    @assert(label.trainable == false)
    @assert( size(var.value) == size(label.value) )
    out = Variable(- label.value .* log.(var.value) .- (1.0 .- label.value) .* log.(1.0 .- var.value))
    if graph.backProp
        function backprop()
            var.delta += out.delta .* ( (1.0 .- label.value) ./ (1.0 .- var.value) .- label.value ./ var.value )
        end
        push!(graph.backward, backprop)
        push!(graph.operator, binaryCrossEntropy)
        push!(var.children, out)
        push!(label.children, out)
        push!(out.parents, var)
        push!(out.parents, label)
    end
    return out
end


function mse(graph::Graph, var::Variable, label::Variable)
    @assert(var.trainable == false)
    @assert(label.trainable == false)
    @assert( size(var.value) == size(label.value) )
    out = Variable( (var.value .- label.value).^2 )
    if graph.backProp
        function backprop()
            var.delta += 2.0 .* (var.value .- label.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, mse)
        push!(var.children, out)
        push!(label.children, out)
        push!(out.parents, var)
        push!(out.parents, label)
    end
    return out
end


function cost(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    row, col = size(var.value)
    out = Variable( sum(var.value) )
    out.delta = 1.0
    if graph.backProp
        function backprop()
            var.delta += var.delta .+ 1.0
        end
        push!(graph.backward, backprop)
        push!(graph.operator, cost)
        push!(var.children, out)
        push!(out.parents, var)
        push!(out.children, nothing)
    end
    return out
end


# 不常用激活函数...


function softplus(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( log.( 1.0 .+ exp.(var.value) ) )
    if graph.backProp
        function backprop()
            var.delta += out.delta ./ (1.0 .+ exp.(-var.value))
        end
        push!(graph.backward, backprop)
        push!(graph.operator, softplus)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function EXP(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( exp.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += exp.(var.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, EXP)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function LOG(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( log.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += out.delta ./ var.value
        end
        push!(graph.backward, backprop)
        push!(graph.operator, LOG)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function ABS(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( abs.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += sign.(var.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, ABS)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


import Base.reshape
function reshape(graph::Graph, var::Variable, newsize)
    @assert(var.trainable == false)
    r,c = size(var.value)
    out = Variable( zeros(r,c) )
    out.value = reshape(var.value, newsize)
    out.delta = reshape(var.delta, newsize)
    return out
end

#
# function dropout(var::Variable, p)
#     # 对网络激活节点进行灭活
#     # 属于本体操作 inplace
#     row, col = size(var.value)
#     randmask = (rand(row, col) .< (1 - p)) ./ (1 - p)
#     var.value .*= randmask
#     var.delta .*= randmask
#     return nothing
# end



function EXP2(graph::Graph, var::Variable)
    # EXP2 represents y = 2^x
    @assert(var.trainable == false)
    out = Variable( exp2.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += log(2.0) .* out.value .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, EXP2)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function EXP10(graph::Graph, var::Variable)
    # EXP10 represents y = 10^x
    @assert(var.trainable == false)
    out = Variable( exp10.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += log(10.0) .* out.value .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, EXP10)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function LOG2(graph::Graph, var::Variable)
    # LOG10 represents y = log2(x)
    @assert(var.trainable == false)
    out = Variable( log10.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += out.delta ./ (log(2.0) .* var.value)
        end
        push!(graph.backward, backprop)
        push!(graph.operator, LOG2)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function LOG10(graph::Graph, var::Variable)
    # LOG10 represents y = log10(x)
    @assert(var.trainable == false)
    out = Variable( log10.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += out.delta ./ (log(10.0) .* var.value)
        end
        push!(graph.backward, backprop)
        push!(graph.operator, LOG10)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function SEC(graph::Graph, var::Variable)
    # SEC represents y = sec(x)
    @assert(var.trainable == false)
    out = Variable( sec.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += out.delta .* out.value .* tan.(var.value)
        end
        push!(graph.backward, backprop)
        push!(graph.operator, SEC)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function SQRT(graph::Graph, var::Variable)
    # SQRT represents y = sqrt(x)
    @assert(var.trainable == false)
    out = Variable( sqrt.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += out.delta ./ (2.0 .* out.value)
        end
        push!(graph.backward, backprop)
        push!(graph.operator, SQRT)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end

# -- tan serials --
function TAN(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( tan.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += (1.0 .+ out.value.^2) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, TAN)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function TAND(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( tand.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += pi/180 .* (1.0 .+ out.value.^2) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, TAND)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function TANH(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( tanh.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += (1.0 .- out.value.^2) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, TANH)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


# -- sin serials --
function SIN(graph::Graph, var::Variable)
    # SIN represents y = sin(x)
    @assert(var.trainable == false)
    out = Variable( sin.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += cos.(var.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, SIN)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function SINC(graph::Graph, var::Variable)
    # SINC represents y = sin(pi*x)/(pi*x)
    @assert(var.trainable == false)
    out = Variable( sinc.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += out.delta .* cosc.(var.value)
        end
        push!(graph.backward, backprop)
        push!(graph.operator, SINC)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function SIND(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( sind.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += pi/180 .* cosd.(var.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, SIND)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function SINPI(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( sinpi.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += pi .* cospi.(var.value) .* out.delta
        end
        push!(graph.backward, backprop)
        push!(graph.operator, SINPI)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end


function INV(graph::Graph, var::Variable)
    @assert(var.trainable == false)
    out = Variable( inv.(var.value) )
    if graph.backProp
        function backprop()
            var.delta += - out.delta ./ var.value.^2
        end
        push!(graph.backward, backprop)
        push!(graph.operator, INV)
        push!(var.children, out)
        push!(out.parents, var)
    end
    return out
end
