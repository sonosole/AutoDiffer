mutable struct rin <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    u::Variable # recurrent weights
    f::Function # activation function
    h           # hidden variable
    function rin(inputSize::Int, hiddenSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        b = zeros(hiddenSize, 1)
        u = randn(hiddenSize, hiddenSize) .* 1e-4
        new(Variable(w,true), Variable(b,true), Variable(u,true), relu, nothing)
    end
    function rin(inputSize::Int, hiddenSize::Int, fn::Function)
        w = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        b = zeros(hiddenSize, 1)
        u = randn(hiddenSize, hiddenSize) .* 1e-4
        new(Variable(w,true), Variable(b,true), Variable(u,true), fn, nothing)
    end
end


mutable struct RIN <: Block
    layernum::Int
    topology::Vector{Int}
    layers::Vector{rin}
    function RIN(topology::Vector{Int})
        layernum = length(topology) - 1
        layers = Vector{rin}(undef, layernum)
        for i = 1:layernum-1
            layers[i] = rin(topology[i], topology[i+1], relu)
        end
        layers[layernum] = rin(topology[layernum], topology[layernum+1], softmax)
        new(layernum, topology, layers)
    end
    function RIN(topology::Vector{Int}, fn::Array{T}) where T
        layernum = length(topology) - 1
        layers = Vector{rin}(undef, layernum)
        for i = 1:layernum
            layers[i] = rin(topology[i], topology[i+1], fn[i])
        end
        new(layernum, topology, layers)
    end
end


function resethidden(model::rin)
    model.h = nothing
end


function resethidden(model::RIN)
    for i = 1:model.layernum
        model.layers[i].h = nothing
    end
end


function forward(model::rin, x::Variable)
    f = model.f  # activition function
    w = model.w  # input's weights
    b = model.b  # input's bias
    u = model.u  # memory's weights
    h = model.h != nothing ? model.h : Variable(zeros(size(w,1),size(x,2)))
    x = f( matAddVec(w*x + u*h + h, b) )
    model.h = x
    return x
end


function forward(model::RIN, input::Variable)
    hlayers = model.layernum
    x = forward(model.layers[1], input)
    for i = 2:hlayers
        x = forward(model.layers[i], x)
    end
    return x
end


function predict(model::rin, x)
    f = model.f        # activition function
    w = model.w.value  # input's weights
    b = model.b.value  # input's bias
    u = model.u.value  # memory's weights
    h = model.h != nothing ? model.h : zeros(size(w,1),size(x,2))
    x = f(w*x + u*h + h .+ b)
    model.h = x
    return x
end


function predict(model::RIN, input)
    x = predict(model.layers[1], input)
    for i = 2:model.layernum
        x = predict(model.layers[i], x)
    end
    return x
end


function weightsof(m::rin)
    weights = Vector(undef,3)
    weights[1] = m.w.value
    weights[2] = m.b.value
    weights[3] = m.u.value
    return weights
end


function weightsof(m::RIN)
    weights = Vector(undef,0)
    for i = 1:m.layernum
        append!(weights, weightsof(m.layers[i]))
    end
    return weights
end


function gradsof(m::rin)
    grads = Vector(undef,3)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    grads[3] = m.u.delta
    return grads
end


function gradsof(m::RIN)
    grads = Vector(undef,0)
    for i = 1:m.layernum
        append!(grads, gradsof(m.layers[i]))
    end
    return grads
end


function zerograds(m::rin)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds(m::RIN)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::rin)
    params = Vector{Variable}(undef,3)
    params[1] = m.w
    params[2] = m.b
    params[3] = m.u
    return params
end


function paramsof(m::RIN)
    params = Vector{Variable}(undef,0)
    for i = 1:m.layernum
        append!(params, paramsof(m.layers[i]))
    end
    return params
end


function nparamsof(m::rin)
    lw = length(m.w)
    lb = length(m.b)
    lu = length(m.u)
    return (lw+lb+lu)
end


function nparamsof(m::RIN)
    num = 0
    for i = 1:m.layernum
        num += nparamsof(m.layers[i])
    end
    return num
end
