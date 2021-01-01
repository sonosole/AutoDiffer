mutable struct rnn <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    u::Variable # recurrent weights
    f::Function # activation function
    h           # hidden variable
    function rnn(inputSize::Int, hiddenSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        b = zeros(hiddenSize, 1)
        u =  eyes(hiddenSize) .* 1e-1 + randn(hiddenSize,hiddenSize) .* 1e-3
        new(Variable(w,true), Variable(b,true), Variable(u,true), relu, nothing)
    end
    function rnn(inputSize::Int, hiddenSize::Int, fn::Function)
        w = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        b = zeros(hiddenSize, 1)
        u =  eyes(hiddenSize) .* 1e-1 + randn(hiddenSize,hiddenSize) .* 1e-3
        new(Variable(w,true), Variable(b,true), Variable(u,true), fn, nothing)
    end
end


mutable struct RNN <: Block
    layernum::Int
    topology::Vector{Int}
    layers::Vector{rnn}
    function RNN(topology::Vector{Int})
        layernum = length(topology) - 1
        layers = Vector{rnn}(undef, layernum)
        for i = 1:layernum-1
            layers[i] = rnn(topology[i], topology[i+1], relu)
        end
        layers[layernum] = rnn(topology[layernum], topology[layernum+1], softmax)
        new(layernum, topology, layers)
    end
    function RNN(topology::Vector{Int}, fn::Array{T}) where T
        layernum = length(topology) - 1
        layers = Vector{rnn}(undef, layernum)
        for i = 1:layernum
            layers[i] = rnn(topology[i], topology[i+1], fn[i])
        end
        new(layernum, topology, layers)
    end
end


function resethidden(model::rnn)
    model.h = nothing
end


function resethidden(model::RNN)
    for i = 1:model.layernum
        model.layers[i].h = nothing
    end
end


function forward(model::rnn, x::Variable)
    f = model.f  # activition function
    w = model.w  # input's weights
    b = model.b  # input's bias
    u = model.u  # memory's weights
    h = model.h != nothing ? model.h : Variable(zeros(size(w,1),size(x,2)))
    x = f(matAddVec(w*x + u*h, b))
    model.h = x
    return x
end


function forward(model::RNN, input::Variable)
    hlayers = model.layernum
    x = forward(model.layers[1], input)
    for i = 2:hlayers
        x = forward(model.layers[i], x)
    end
    return x
end


function predict(model::rnn, x)
    f = model.f        # activition function
    w = model.w.value  # input's weights
    b = model.b.value  # input's bias
    u = model.u.value  # memory's weights
    h = model.h != nothing ? model.h : zeros(size(w,1),size(x,2))
    x = f(w*x + u*h .+ b)
    model.h = x
    return x
end


function predict(model::RNN, input)
    x = predict(model.layers[1], input)
    for i = 2:model.layernum
        x = predict(model.layers[i], x)
    end
    return x
end


function weightsof(m::rnn)
    weights = Vector(undef,3)
    weights[1] = m.w.value
    weights[2] = m.b.value
    weights[3] = m.u.value
    return weights
end


function weightsof(m::RNN)
    weights = Vector(undef,0)
    for i = 1:m.layernum
        append!(weights, weightsof(m.layers[i]))
    end
    return weights
end


function gradsof(m::rnn)
    grads = Vector(undef,3)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    grads[3] = m.u.delta
    return grads
end


function gradsof(m::RNN)
    grads = Vector(undef,0)
    for i = 1:m.layernum
        append!(grads, gradsof(m.layers[i]))
    end
    return grads
end


function zerograds(m::rnn)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds(m::RNN)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::rnn)
    params = Vector{Variable}(undef,3)
    params[1] = m.w
    params[2] = m.b
    params[3] = m.u
    return params
end


function paramsof(m::RNN)
    params = Vector{Variable}(undef,0)
    for i = 1:m.layernum
        append!(params, paramsof(m.layers[i]))
    end
    return params
end


function nparamsof(m::rnn)
    lw = length(m.w)
    lb = length(m.b)
    lu = length(m.u)
    return (lw+lb+lu)
end


function nparamsof(m::RNN)
    num = 0
    for i = 1:m.layernum
        num += nparamsof(m.layers[i])
    end
    return num
end
