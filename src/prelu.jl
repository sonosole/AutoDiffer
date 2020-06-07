function ZEROS(vecsize::Int)
    return zeros(vecsize,1)
end


function ONES(vecsize::Int)
    return ones(vecsize,1)
end


function CONST(vecsize::Int; amp=1.0)
    return amp*ones(vecsize,1)
end


function RANDN(vecsize::Int; meanv=0.0, stdv=1.0)
    return stdv*randn(vecsize,1) .+ meanv
end


mutable struct prelu <: Block
    p::Variable # hidden summations weights
    function parmsrelu(vecsize::Int; initFn::Function=ZEROS)
        p = initFn(vecsize)
        new(Variable(p,true))
    end
end


function paramsof(m::prelu)
    params = Vector(undef,1)
    params[1] = m.p
    return params
end


function forward(model::prelu, input::Variable)
    temp = input.value .* model.p
    mask = input.value .< temp
    out = Variable(max.(temp, input.value), input.trainable)
    if input.trainable
        function preluBackward()
            temp.delta += ...
            var.delta += ...out.delta
        end
        push!(graph.backward, preluBackward)
    end
    return out
end
