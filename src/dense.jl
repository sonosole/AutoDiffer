mutable struct dense <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    f::Function # activation function
    function dense(inputSize::Int, hiddenSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 6 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), relu)
    end
    function dense(inputSize::Int, hiddenSize::Int, fn::Function)
        w = randn(hiddenSize, inputSize) .* sqrt( 6 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), fn)
    end
end


mutable struct MLP <: Block
    layernum::Int
    topology::Vector{Int}
    layers::Vector{dense}
    function MLP(topology::Vector{Int})
        layernum = length(topology) - 1
        layers = Vector{dense}(undef, layernum)
        for i = 1:layernum-1
            layers[i] = dense(topology[i], topology[i+1], relu)
        end
        layers[layernum] = dense(topology[layernum], topology[layernum+1], softmax)
        new(layernum, topology, layers)
    end
    function MLP(topology::Vector{Int}, fn::Vector{T}) where T
        layernum = length(topology) - 1
        layers = Vector{dense}(undef, layernum)
        for i = 1:layernum
            layers[i] = dense(topology[i], topology[i+1], fn[i])
        end
        new(layernum, topology, layers)
    end
end


function forward(model::dense, input::Variable)
    f = model.f
    w = model.w
    b = model.b
    x = f( matAddVec(w * input, b) )
    return x
end


function forward(model::MLP, input::Variable)
    x = forward(model.layers[1], input)
    for i = 2:model.layernum
        x = forward(model.layers[i], x)
    end
    return x
end


function predict(model::dense, input)
    f = model.f
    w = model.w.value
    b = model.b.value
    x = f(w * input .+ b)
    return x
end


function predict(model::MLP, input)
    x = predict(model.layers[1], input)
    for i = 2:model.layernum
        x = predict(model.layers[i], x)
    end
    return x
end


function weightsof(m::dense)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function weightsof(m::MLP)
    weights = Vector(undef,0)
    for i = 1:m.layernum
        append!(weights, weightsof(m.layers[i]))
    end
    return weights
end


function gradsof(m::dense)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function gradsof(m::MLP)
    grads = Vector(undef,0)
    for i = 1:m.layernum
        append!(grads, gradsof(m.layers[i]))
    end
    return grads
end


function zerograds(m::dense)
    for v in gradsof(m)
        v .= 0.0
    end
end


function zerograds(m::MLP)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::dense)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function paramsof(m::MLP)
    params = Vector{Variable}(undef,0)
    for i = 1:m.layernum
        append!(params, paramsof(m.layers[i]))
    end
    return params
end


function nparamsof(m::dense)
    i,j = size(m.w.value)
    k = size(m.b.value,1)
    return (i*j+k)
end


function nparamsof(m::MLP)
    num = 0
    for i = 1:m.layernum
        num += nparamsof(m.layers[i])
    end
    return num
end
