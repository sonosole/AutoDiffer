include("./AutoDiffer.jl")


mutable struct dense <: Block
    w # input to hidden weights
    b # bias of hidden units
    f # function type
    function dense(hiddenSize::Int, inputSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 6 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), relu)
    end
    function dense(hiddenSize::Int, inputSize::Int, fn::Function)
        w = randn(hiddenSize, inputSize) .* sqrt( 6 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true), fn)
    end
end


mutable struct MLP <: Block
    layernum::Int
    topology::Array{Int,1}
    parameters::Vector

    function MLP(topology::Array{Int,1})
        layernum = length(topology)
        parameters = Vector(undef, layernum-1)
        for i = 1:layernum-2
            parameters[i] = dense(topology[i+1], topology[i], relu)
        end
        parameters[layernum-1] = dense(topology[layernum], topology[layernum-1], softmax)
        new(layernum, topology, parameters)
    end

    function MLP(topology::Array{Int,1}, fn)
        layernum = length(topology)
        parameters = Vector(undef, layernum-1)
        for i = 1:layernum-1
            parameters[i] = dense(topology[i+1], topology[i], fn[i])
        end
        new(layernum, topology, parameters)
    end
end


function weightsof(m::dense)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function weightsof(m::MLP)
    hlayers = m.layernum-1
    weights = Vector(undef,0)
    for i = 1:hlayers
        append!(weights, weightsof(m.parameters[i]))
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
    hlayers = m.layernum-1
    grads = Vector(undef,0)
    for i = 1:hlayers
        append!(grads, gradsof(m.parameters[i]))
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
    params = Vector(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function paramsof(m::MLP)
    hlayers = m.layernum-1
    params = Vector(undef,0)
    for i = 1:hlayers
        append!(params, paramsof(m.parameters[i]))
    end
    return params
end


function forward(model::MLP, input::Variable)
    f = model.parameters[1].f
    w = model.parameters[1].w
    b = model.parameters[1].b
    x = f( matAddVec(w * input, b) )
    for i = 2:model.layernum-1
        f = model.parameters[i].f
        w = model.parameters[i].w
        b = model.parameters[i].b
        x = f( matAddVec(w * x, b) )
    end
    return x
end


function forward(model::dense, input::Variable)
    f = model.f
    w = model.w
    b = model.b
    x = f( matAddVec(w * input, b) )
    return x
end


function predicate(model::MLP, input)
    f = model.parameters[1].f
    w = model.parameters[1].w.value
    b = model.parameters[1].b.value
    x = f(w * input .+ b)
    for i = 2:model.layernum-1
        f = model.parameters[i].f
        w = model.parameters[i].w.value
        b = model.parameters[i].b.value
        x = f(w*x .+ b)
    end
    return x
end


function predicate(model::dense, input)
    f = model.f
    w = model.w.value
    b = model.b.value
    x = f(w * input .+ b)
    return x
end


#
