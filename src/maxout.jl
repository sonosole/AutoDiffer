mutable struct maxout <: Block
    h::Int
    k::Int
    w::Variable # input to middle hidden weights
    b::Variable # bias of middle hidden units
    function maxout(inputSize::Int, hiddenSize::Int; k::Int=2)
        @assert (k>=2) "# of affine layers should no less than 2"
        d = hiddenSize * k
        w = randn(d, inputSize) .* sqrt( 1 / d )
        b = zeros(d, 1)
        new(hiddenSize, k, Variable(w,true), Variable(b,true))
    end
end


function paramsof(m::maxout)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function nparamsof(m::maxout)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


function forward(model::maxout, x::Variable)
    h = model.h
    k = model.k
    w = model.w
    b = model.b
    c = size(x, 2)
    x = matAddVec(w * x, b)         # dim=(h*k, c)
    temp = reshape(x.value, h,k,c)  # dim=(h,k,c)
    maxv = maximum(temp, dims=2)    # dim=(h,1,c)
    mask = temp .== maxv            # dim=(h,k,c)
    out  = Variable(reshape(maxv, h,c), x.trainable)
    if x.trainable
        function maxoutBackward()
            x.delta += reshape(mask .* reshape(out.delta, h,1,c), h*k,c)
        end
        push!(graph.backward, maxoutBackward)
    end
    return out
end


function predict(model::maxout, x)
    h = model.h
    k = model.k
    w = model.w.value
    b = model.b.value
    c = size(x, 2)
    x = w * x .+ b                  # dim=(h*k, c)
    temp = reshape(x, h,k,c)        # dim=(h,k,c)
    maxv = maximum(temp, dims=2)    # dim=(h,1,c)
    out  = reshape(maxv, h,c)       # dim=(h,  c)
    return out
end


function nparamsof(m::maxout)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end
