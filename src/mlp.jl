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
    parameter::Vector

    function MLP(topology::Array{Int,1})
        layernum  = length(topology)
        parameter = Vector(undef, layernum-1)
        for i = 1:layernum-2
            parameter[i] = dense(topology[i+1], topology[i], relu)
        end
        parameter[layernum-1] = dense(topology[layernum], topology[layernum-1], softmax)
        new(layernum, topology, parameter)
    end

    function MLP(topology::Array{Int,1}, fn)
        layernum  = length(topology)
        parameter = Vector(undef, layernum-1)
        for i = 1:layernum-1
            parameter[i] = dense(topology[i+1], topology[i], fn[i])
        end
        new(layernum, topology, parameter)
    end
end

function weightsof(m::dense)
    # weights parameters
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function weightsof(m::MLP)
    hlayers = m.layernum-1
    weights = Vector(undef,0)
    for i = 1:hlayers
        append!(weights, weightsof(m.parameter[i]))
    end
    return weights
end


function paramsof(m::dense)
    # variable parameters
    params = Vector(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function paramsof(m::MLP)
    hlayers = m.layernum-1
    params = Vector(undef,0)
    for i = 1:hlayers
        append!(params, paramsof(m.parameter[i]))
    end
    return params
end


function gradof(m::dense)
    grad = Vector(undef, 2)
    grad[1] = m.w.delta
    grad[2] = m.b.delta
    return grad
end


function gradof(m::MLP)
    hlayers = m.layernum-1
    grad = Vector(undef,0)
    for i = 1:hlayers
        append!(grad, gradof(m.parameter[i]))
    end
    return grad
end


function zerograd(m::dense)
    for v in gradof(m)
        v .= zero(v)
    end
end


function zerograd(m::MLP)
    for v in gradof(m)
        v .= zero(v)
    end
end


function forward(model::MLP, input::Variable)
    f = model.parameter[1].f
    w = model.parameter[1].w
    b = model.parameter[1].b
    x = f( matAddVec(w * input, b) )
    for i = 2:model.layernum-1
        f = model.parameter[i].f
        w = model.parameter[i].w
        b = model.parameter[i].b
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
    f = model.parameter[1].f
    w = model.parameter[1].w.value
    b = model.parameter[1].b.value
    x = f.(w * input .+ b)
    for i = 2:model.layernum-1
        f = model.parameter[i].f
        w = model.parameter[i].w.value
        b = model.parameter[i].b.value
        x = f.(w*x .+ b)
    end
    return x
end


function predicate(model::dense, input)
    f = model.f
    w = model.w.value
    b = model.b.value
    x = f.(w * input .+ b)
    return x
end


#
