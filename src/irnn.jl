include("./ad.jl")

mutable struct irnn <: Block
    w # input to hidden weights
    b # bias of hidden units
    u # recurrent weights
    function irnn(hiddenSize::Int, inputSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 3 / (hiddenSize + inputSize) )
        b = zeros(hiddenSize, 1)
        u = zeros(hiddenSize, 1)
        new(w, b, u)
    end
end


mutable struct iRNN <: Block
    layernum::Int
    topology::Array{Int,1}
    operator::Array{Function,1}
    parameter::Vector

    function iRNN(topology::Array{Int,1})
        layernum  = length(topology)
        parameter = Vector(undef, layernum-1)
        operator  = Vector(undef, layernum-1)
        for i = 1:layernum-2
            parameter[i] = irnn(topology[i+1], topology[i])
            operator[i]  = relu
        end
        parameter[layernum-1] = irnn(topology[layernum], topology[layernum-1])
        operator[layernum-1]  = softmax
        new(layernum, topology, operator, parameter)
    end

    function iRNN(topology::Array{Int,1}, operator::Array{Function,1})
        layernum  = length(topology)
        parameter = Vector(undef, layernum-1)
        for i = 1:layernum-1
            parameter[i] = irnn(topology[i+1], topology[i])
        end
        new(layernum, topology, operator, parameter)
    end
end


function forward(graph::Graph, model::iRNN, input)
    @assert length( size(input) ) == 3 "shape(input) = [dims, bathSize, seqLen]"
    dims, bathSize, seqLen = size(input)
    # firs time step
    i = Variable(input[:,:,1], false)
    w = Variable(model.parameter[1].w, true)
    b = Variable(model.parameter[1].b, true)
    u = Variable(model.parameter[1].u, true)
    f = model.operator[1]
    x = f(graph, matAddVec(graph, matMul(graph, w, i), b))
    references = Vector(undef, 0)
    push!(references, w)
    push!(references, b)
    push!(references, u)
    for i = 2:model.layernum-1
        w = Variable(model.parameter[i].w, true)
        b = Variable(model.parameter[i].b, true)
        u = Variable(model.parameter[i].u, true)
        f = model.operator[i]
        x = f(graph, matAddVec(graph, matMul(graph, w, x), b))
        push!(references, w)
        push!(references, b)
        push!(references, u)
    end

    for t = 2:seqLen
        i = Variable(input[:,:,t], false)
        w = Variable(model.parameter[1].w, true)
        b = Variable(model.parameter[1].b, true)
        u = Variable(model.parameter[1].u, true)
        f = model.operator[1]
        x = f(graph, matAddVec(graph, matMul(graph, w, i), b))
    end

    return x, references
end


g = Graph(true)
ind = iRNN([2,3,4])
input = randn(2,10,100)

forward(g, ind, input)
