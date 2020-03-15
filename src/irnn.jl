include("./AutoDiffer.jl")


mutable struct irnn <: Block
    w # input to hidden weights
    b # bias of hidden units
    u # recurrent weights
    f # function type
    function irnn(hiddenSize::Int, inputSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 3 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        u = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), Variable(u,true), relu)
    end
    function irnn(hiddenSize::Int, inputSize::Int, fn::Function)
        w = randn(hiddenSize, inputSize) .* sqrt( 3 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        u = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), Variable(u,true), fn)
    end
end


mutable struct iRNN <: Block
    layernum::Int
    topology::Array{Int,1}
    parameters::Vector

    function iRNN(topology::Array{Int,1})
        layernum = length(topology)
        parameters = Vector(undef, layernum-1)
        for i = 1:layernum-1
            parameters[i] = irnn(topology[i+1], topology[i])
        end
        new(layernum, topology, parameters)
    end

    function iRNN(topology::Array{Int,1}, fn)
        layernum = length(topology)
        parameters = Vector(undef, layernum-1)
        for i = 1:layernum-1
            parameters[i] = irnn(topology[i+1], topology[i], fn[i])
        end
        new(layernum, topology, parameters)
    end
end


function weightsof(m::irnn)
    weights = Vector(undef,3)
    weights[1] = m.w.value
    weights[2] = m.b.value
    weights[3] = m.u.value
    return weights
end


function weightsof(m::iRNN)
    hlayers = m.layernum-1
    weights = Vector(undef,0)
    for i = 1:hlayers
        append!(weights, weightsof(m.parameters[i]))
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


function gradsof(m::iRNN)
    hlayers = m.layernum-1
    grads = Vector(undef,0)
    for i = 1:hlayers
        append!(grads, gradsof(m.parameters[i]))
    end
    return grads
end


function zerograds(m::irnn)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds(m::iRNN)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::irnn)
    params = Vector(undef,3)
    params[1] = m.w
    params[2] = m.b
    params[3] = m.u
    return params
end


function paramsof(m::iRNN)
    hlayers = m.layernum-1
    params = Vector(undef,0)
    for i = 1:hlayers
        append!(params, paramsof(m.parameters[i]))
    end
    return params
end


function forward(model::iRNN, input::Variable)
    # @assert length( size(input.value) ) == 3 "shape(input) = [dims, bathSize, seqLen]"
    dims, bathSize, seqLen = size(input.value)
    # first timestep
    hlayers = model.layernum - 1
    hiddens = Vector(undef, hlayers)
    outputs = Vector(undef, seqLen)

    f = model.parameters[1].f
    w = model.parameters[1].w
    b = model.parameters[1].b
    x = f(matAddVec(w*input[:,:,1], b))
    hidden[1] = x
    for i = 2:hlayers
        f = model.parameters[i].f  # activition function
        w = model.parameters[i].w  # input's weights
        b = model.parameters[i].b  # input's bias
        x = f(matAddVec(w*x, b))  # output
        hidden[i] = x             # as hidden
    end
    outputs[1] = x

    # other timesteps
    for t = 2:seqLen
        x, hidden[1] = forward(model.parameters[1], input[:,:,t], hidden[1])
        for i = 2:hlayers
            x, hidden[i] = forward(model.parameters[i], x, hidden[i])
        end
        outputs[t] = x
    end
    return outputs
end


function forward(model::irnn, input::Variable, hidden::Variable)
    f = model.f  # activition function
    w = model.w  # input's weights
    b = model.b  # input's bias
    u = model.u  # memory's weights
    m = matMulVec(hidden, u)         # memory
    x = f(matAddVec(w*input + m, b)) # output
    return x, x
end


function predicate(model::irnn, input, hidden)
    f = model.f        # activition function
    w = model.w.value  # input's weights
    b = model.b.value  # input's bias
    u = model.u.value  # memory's weights
    m = matMulVec(hidden, u)         # memory
    x = f(matAddVec(w*input + m, b)) # output
    return x, x
end
