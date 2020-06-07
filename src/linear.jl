mutable struct linear <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    function linear(inputSize::Int, hiddenSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 1 / hiddenSize )
        b = zeros(hiddenSize, 1)
        new(Variable(w,true), Variable(b,true))
    end
end


function weightsof(m::linear)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::linear)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds(m::linear)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::linear)
    params = Vector(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function nparamsof(m::linear)
    i,j = size(m.w.value)
    k = size(m.b.value,1)
    return (i*j+k)
end


function forward(model::linear, input::Variable)
    w = model.w
    b = model.b
    x = matAddVec(w * input, b)
    return x
end


function predict(model::linear, input)
    w = model.w.value
    b = model.b.value
    x = w * input .+ b
    return x
end
