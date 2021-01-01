mutable struct linear <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    function linear(inputSize::Int, hiddenSize::Int)
        w = randn(hiddenSize, inputSize) .* sqrt( 1 / hiddenSize )
        b = randn(hiddenSize, 1) .* sqrt( 1 / hiddenSize )
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
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


function forward(model::linear, x::Variable)
    w = model.w
    b = model.b
    return matAddVec(w * x, b)
end


function predict(model::linear, x)
    w = model.w.value
    b = model.b.value
    return (w * x .+ b)
end
