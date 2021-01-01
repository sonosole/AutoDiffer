mutable struct P1Relu <: Block
    p::Variable
    function P1Relu(initvalue=0.1)
        new(Variable(initvalue*ones(1,1),true))
    end
end


function paramsof(m::P1Relu)
    params = Vector(undef,1)
    params[1] = m.p
    return params
end


function forward(model::P1Relu, input::Variable)
    r, c  = size(input)
    slope = model.p.value
    tempv = input.value .* slope
    mask1 = input.value .> tempv
    mask2 = .!mask1
    out   = Variable(max.(tempv, input.value), input.trainable)
    if input.trainable
        function P1ReluBackward()
            model.p.delta .+= sum(out.delta .* mask2 .* input.value) / (r*c)
            input.delta += out.delta .* (mask1 + slope .* mask2)
        end
        push!(graph.backward, P1ReluBackward)
    end
    return out
end


function predict(m::P1Relu, input)
    return max.(m.p.value .* input, input)
end
