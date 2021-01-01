mutable struct conv_1xd <: Block
    w::Variable # input to hidden weights
    b::Variable # bias of hidden units
    k::Int      # kernel size
    s::Int      # stride size
    p::Int      # padding size
    function conv_1xd(ichannels::Int, ochannels::Int, kernel::Int;
                      stride::Int = 1,
                      padding::Int = 0)
        filterSize = ichannels * kernel
        w = randn(ochannels, filterSize) .* sqrt( 2 / filterSize )
        b = randn(ochannels, 1) .* sqrt( 2 / filterSize )
        new(Variable(w,true), Variable(b,true), kernel, stride, padding)
    end
end


function weightsof(m::conv_1xd)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::conv_1xd)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds(m::conv_1xd)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::conv_1xd)
    params = Vector(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function nparamsof(m::conv_1xd)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


function in2col(var::Variable, kernel::Int, stride::Int)
    # from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    # in which cols = (width – kernel + 1) * batchsize
    (ichannels,width,batchsize) = size(var)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    out  = Variable(zeros(rows, cols), var.trainable)
    k    = 1
    for b = 1:batchsize
        start = 1
        final = kernel
        for s = 1:step
            out.value[:,k] = reshape(var.value[:,start:final,b], (rows,1))
            start += stride
            final += stride
            k     += 1
        end
    end
    if var.trainable
        function in2colBackward()
            k = 1
            for b = 1:batchsize
                start = 1
                final = kernel
                for s = 1:step
                    var.delta[:,start:final,b] += reshape(out.delta[:,k], (ichannels, kernel))
                    start += stride
                    final += stride
                    k     += 1
                end
            end
        end
        push!(graph.backward, in2colBackward)
    end
    return out
end


function col2out(x::Variable, batchsize::Int)
    # from (ochannels,width*batchsize) to (ochannels,width,batchsize)
    (ochannels,cols) = size(x)
    width = div(cols, batchsize)
    return reshape(x, (ochannels,width,batchsize))
end


function in2col(var::Array, kernel::Int, stride::Int)
    # from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    # in which cols = (width – kernel + 1) * batchsize
    (ichannels,width,batchsize) = size(var)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    out  = zeros(rows, cols)
    k    = 1
    for b = 1:batchsize
        start = 1
        final = kernel
        for s = 1:step
            out.value[:,k] = reshape(var.value[:,start:final,b], (rows,1))
            start += stride
            final += stride
            k     += 1
        end
    end
    return out
end


function col2out(x::Array, batchsize::Int)
    # from (ochannels,width*batchsize) to (ochannels,width,batchsize)
    (ochannels,cols) = size(x)
    width = div(cols, batchsize)
    return reshape(x, (ochannels,width,batchsize))
end


function forward(model::conv_1xd, x::Variable)
    # size(x) == (ichannels,width,batchsize)
    @assert ndims(x)==3 "input shape is of (ichannels,width,batchsize)"
    batchsize = size(x,3)
    w = model.w
    b = model.b
    x = in2col(x, model.k, model.s)
    x = matAddVec(w * x, b)
    return col2out(x, batchsize)
end


function predict(model::conv_1xd, x::Array)
    # size(x) == (ichannels,width,batchsize)
    @assert ndims(x)==3 "input shape is of (ichannels,width,batchsize)"
    batchsize = size(x,3)
    w = model.w.value
    b = model.b.value
    x = in2col(x, model.k, model.s)
    x = matAddVec(w * x .+ b)
    return col2out(x, batchsize)
end
