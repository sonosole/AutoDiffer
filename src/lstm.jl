mutable struct lstm <: Block
    # input control gate params
    wi::Variable
    ui::Variable
    bi::Variable
    # forget control gate params
    wf::Variable
    uf::Variable
    bf::Variable
    # out control gate params
    wo::Variable
    uo::Variable
    bo::Variable
    # new cell info params
    wc::Variable
    uc::Variable
    bc::Variable
    h  # hidden variable
    c  # cell variable
    function lstm(inputSize::Int, hiddenSize::Int)
        wi = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        ui = eyes(hiddenSize)
        bi = zeros(hiddenSize, 1)

        wf = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        uf =  eyes(hiddenSize)
        bf =  ones(hiddenSize, 1)

        wo = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        uo = eyes(hiddenSize)
        bo = zeros(hiddenSize, 1)

        wc = randn(hiddenSize, inputSize) .* sqrt( 2 / inputSize )
        uc = eyes(hiddenSize)
        bc = zeros(hiddenSize, 1)

        new(Variable(wi,true), Variable(ui,true), Variable(bi,true),
            Variable(wf,true), Variable(uf,true), Variable(bf,true),
            Variable(wo,true), Variable(uo,true), Variable(bo,true),
            Variable(wc,true), Variable(uc,true), Variable(bc,true),nothing, nothing)
    end
end


mutable struct LSTM <: Block
    layernum::Int
    topology::Vector{Int}
    layers::Vector{lstm}
    function LSTM(topology::Vector{Int})
        layernum = length(topology) - 1
        layers = Vector{lstm}(undef, layernum)
        for i = 1:layernum-1
            layers[i] = lstm(topology[i], topology[i+1])
        end
        layers[layernum] = lstm(topology[layernum], topology[layernum+1])
        new(layernum, topology, layers)
    end
end


function resethidden(model::lstm)
    model.h = nothing
    model.c = nothing
end


function resethidden(model::LSTM)
    for i = 1:model.layernum
        model.layers[i].h = nothing
        model.layers[i].c = nothing
    end
end


function forward(model::lstm, x::Variable)
    wi = model.wi
    ui = model.ui
    bi = model.bi

    wf = model.wf
    uf = model.uf
    bf = model.bf

    wo = model.wo
    uo = model.uo
    bo = model.bo

    wc = model.wc
    uc = model.uc
    bc = model.bc

    h = model.h != nothing ? model.h : Variable(zeros(size(wi,1),size(x,2)))
    c = model.c != nothing ? model.c : Variable(zeros(size(wc,1),size(x,2)))

    z = tanh(    matAddVec(wc * x + uc * h, bc) )
    i = sigmoid( matAddVec(wi * x + ui * h, bi) )
    f = sigmoid( matAddVec(wf * x + uf * h, bf) )
    o = sigmoid( matAddVec(wo * x + uo * h, bo) )
    c = dotMul(f, c) + dotMul(i, z)
    h = dotMul(o, c)

    model.c = c
    model.h = h

    return h
end


function forward(model::LSTM, input::Variable)
    hlayers = model.layernum
    x = forward(model.layers[1], input)
    for i = 2:hlayers
        x = forward(model.layers[i], x)
    end
    return x
end


function predict(model::lstm, x)
    wi = model.wi.value
    ui = model.ui.value
    bi = model.bi.value

    wf = model.wf.value
    uf = model.uf.value
    bf = model.bf.value

    wo = model.wo.value
    uo = model.uo.value
    bo = model.bo.value

    wc = model.wc.value
    uc = model.uc.value
    bc = model.bc.value

    h = model.h != nothing ? model.h : zeros(size(wi,1),size(x,2))
    c = model.c != nothing ? model.c : zeros(size(wc,1),size(x,2))

    z = tanh(    wc * x + uc * h .+ bc )
    i = sigmoid( wi * x + ui * h .+ bi )
    f = sigmoid( wf * x + uf * h .+ bf )
    o = sigmoid( wo * x + uo * h .+ bo )
    c = f .* c + i .* z
    h = o .* c

    model.c = c
    model.h = h

    return h
end


function predict(model::LSTM, input)
    x = predict(model.layers[1], input)
    for i = 2:model.layernum
        x = predict(model.layers[i], x)
    end
    return x
end


function weightsof(m::lstm)
    weights = Vector{Variable}(undef,12)
    weights[1] = m.wi.value
    weights[2] = m.ui.value
    weights[3] = m.bi.value
    weights[4] = m.wf.value
    weights[5] = m.uf.value
    weights[6] = m.bf.value
    weights[7] = m.wo.value
    weights[8] = m.uo.value
    weights[9] = m.bo.value
    weights[10] = m.wc.value
    weights[11] = m.uc.value
    weights[12] = m.bc.value
    return weights
end

function weightsof(m::LSTM)
    weights = Vector(undef,0)
    for i = 1:m.layernum
        append!(weights, weightsof(m.layers[i]))
    end
    return weights
end


function gradsof(m::lstm)
    grads = Vector{Variable}(undef,12)
    grads[1] = m.wi.delta
    grads[2] = m.ui.delta
    grads[3] = m.bi.delta
    grads[4] = m.wf.delta
    grads[5] = m.uf.delta
    grads[6] = m.bf.delta
    grads[7] = m.wo.delta
    grads[8] = m.uo.delta
    grads[9] = m.bo.delta
    grads[10] = m.wc.delta
    grads[11] = m.uc.delta
    grads[12] = m.bc.delta
    return grads
end


function gradsof(m::LSTM)
    grads = Vector(undef,0)
    for i = 1:m.layernum
        append!(grads, gradsof(m.layers[i]))
    end
    return grads
end


function zerograds(m::lstm)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function zerograds(m::LSTM)
    for v in gradsof(m)
        v .= zero(v)
    end
end


function paramsof(m::lstm)
    params = Vector{Variable}(undef,12)
    params[1] = m.wi
    params[2] = m.ui
    params[3] = m.bi
    params[4] = m.wf
    params[5] = m.uf
    params[6] = m.bf
    params[7] = m.wo
    params[8] = m.uo
    params[9] = m.bo
    params[10] = m.wc
    params[11] = m.uc
    params[12] = m.bc
    return params
end


function paramsof(m::LSTM)
    params = Vector{Variable}(undef,0)
    for i = 1:m.layernum
        append!(params, paramsof(m.layers[i]))
    end
    return params
end


function nparamsof(m::lstm)
    lw = length(m.wi)
    lu = length(m.ui)
    lb = length(m.bi)
    return (lw+lu+lb)*4
end


function nparamsof(m::LSTM)
    num = 0
    for i = 1:m.layernum
        num += nparamsof(m.layers[i])
    end
    return num
end
