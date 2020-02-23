include("./AutoDiffer.jl")


mutable struct dense <: Block
    w # input to hidden weights
    b # bias of hidden units
    function dense(hiddenSize::Int, inputSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 6 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        new(w, b)
    end
end


mutable struct MLP <: Block
    layernum::Int
    topology::Array{Int,1}
    operator::Array{Function,1}
    parameter::Vector

    function MLP(topology::Array{Int,1})
        layernum  = length(topology)
        parameter = Vector(undef, layernum-1)
        operator  = Vector(undef, layernum-1)
        for i = 1:layernum-2
            parameter[i] = dense(topology[i+1], topology[i])
            operator[i]  = relu
        end
        parameter[layernum-1] = dense(topology[layernum], topology[layernum-1])
        operator[layernum-1]  = softmax
        new(layernum, topology, operator, parameter)
    end

    function MLP(topology::Array{Int,1}, operation::Array{Function,1})
        layernum  = length(topology)
        parameter = Vector(undef, layernum-1)
        operator  = Vector(undef, layernum-1)
        for i = 1:layernum-1
            parameter[i] = dense(topology[i+1], topology[i])
            operator[i]  = operation[i]
        end
        new(layernum, topology, operator, parameter)
    end
end


function forward(graph::Graph, model::MLP, input)
    references = Vector(undef, 0)
    f = model.operator[1]
    i = Variable(input, false)
    w = Variable(model.parameter[1].w, true)
    b = Variable(model.parameter[1].b, true)
    x = f(graph, matAddVec(graph, matMul(graph, w, i), b))
    push!(references, w)
    push!(references, b)
    for k = 2:model.layernum-1
        f = model.operator[k]
        w = Variable(model.parameter[k].w, true)
        b = Variable(model.parameter[k].b, true)
        x = f(graph, matAddVec(graph, matMul(graph, w, x), b))
        push!(references, w)
        push!(references, b)
    end
    return x, references
end


function predicate(model::MLP, input)
    g = Graph(false)
    f = model.operator[1]
    i = Variable(input, false)
    w = Variable(model.parameter[1].w, true)
    b = Variable(model.parameter[1].b, true)
    x = f(g, matAddVec(g, matMul(g, w, i), b))
    for k = 2:model.layernum-1
        f = model.operator[k]
        w = Variable(model.parameter[k].w, true)
        b = Variable(model.parameter[k].b, true)
        x = f(g, matAddVec(g, matMul(g, w, x), b))
    end
    return x
end
