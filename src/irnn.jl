mutable struct irnn <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    u::Variable # recurrent weights
    f::Function # activation function
    h           # hidden variable
    function irnn(inputSize::Int, hiddenSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 6 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        u = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), Variable(u,true), relu, nothing)
    end
    function irnn(inputSize::Int, hiddenSize::Int, fn::Function)
        w = randn(hiddenSize, inputSize) .* sqrt( 6 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        u = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), Variable(u,true), fn, nothing)
    end
end


mutable struct IRNN <: Block
    layernum::Int
    topology::Vector{Int}
    layers::Vector{irnn}
    function IRNN(topology::Vector{Int})
        layernum = length(topology) - 1
        layers = Vector{irnn}(undef, layernum)
        for i = 1:layernum-1
            layers[i] = irnn(topology[i], topology[i+1], relu)
        end
        layers[layernum] = irnn(topology[layernum], topology[layernum+1], softmax)
        new(layernum, topology, layers)
    end

    function IRNN(topology::Vector{Int}, fn::Array{T}) where T
        layernum = length(topology) - 1
        layers = Vector{irnn}(undef, layernum)
        for i = 1:layernum
            layers[i] = irnn(topology[i], topology[i+1], fn[i])
        end
        new(layernum, topology, layers)
    end
end


function resethidden(model::irnn)
    model.h = nothing
end


function resethidden(model::IRNN)
    for i = 1:model.layernum
        model.layers[i].h = nothing
    end
end


function forward(model::irnn, input::Variable)
    f = model.f  # activition function
    w = model.w  # input's weights
    b = model.b  # input's bias
    u = model.u  # memory's weights
    h = model.h != nothing ? model.h : Variable(zeros(size(w.value,1),size(input.value,2)))
    x = f(matAddVec(w*input + matMulVec(h, u), b))
    model.h = x
    return x
end


function forward(model::IRNN, input::Variable)
    hlayers = model.layernum
    x = forward(model.layers[1], input)
    for i = 2:hlayers
        x = forward(model.layers[i], x)
    end
    return x
end


function predict(model::irnn, input)
    f = model.f        # activition function
    w = model.w.value  # input's weights
    b = model.b.value  # input's bias
    u = model.u.value  # memory's weights
    h = model.h != nothing ? model.h : zeros(size(w,1),size(input,2))
    x = f(w*input + h .* u .+ b)
    model.h = x
    return x
end


function predict(model::IRNN, input)
    x = predict(model.layers[1], input)
    for i = 2:model.layernum
        x = predict(model.layers[i], x)
    end
    return x
end


function weightsof(m::irnn)
    weights = Vector(undef,3)
    weights[1] = m.w.value
    weights[2] = m.b.value
    weights[3] = m.u.value
    return weights
end


function weightsof(m::IRNN)
    weights = Vector(undef,0)
    for i = 1:m.layernum
        append!(weights, weightsof(m.layers[i]))
    end
    return weights
end


function gradsof(m::irnn)
    grads = Vector(undef,3)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    grads[3] = m.u.delta
    return grads
end


function gradsof(m::IRNN)
    grads = Vector(undef,0)
    for i = 1:m.layernum
        append!(grads, gradsof(m.layers[i]))
    end
    return grads
end


function zerograds(m::irnn)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds(m::IRNN)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::irnn)
    params = Vector{Variable}(undef,3)
    params[1] = m.w
    params[2] = m.b
    params[3] = m.u
    return params
end


function paramsof(m::IRNN)
    params = Vector{Variable}(undef,0)
    for i = 1:m.layernum
        append!(params, paramsof(m.layers[i]))
    end
    return params
end


function nparamsof(m::irnn)
    i,j = size(m.w.value)
    k = size(m.b.value,1)
    return (i*j+2*k)
end


function nparamsof(m::IRNN)
    num = 0
    for i = 1:m.layernum
        num += nparamsof(m.layers[i])
    end
    return num
end
