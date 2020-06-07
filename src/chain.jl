mutable struct Chain
    blocknum::Int
    blocks::Vector
    function Chain(sequence::Vector)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocknum, blocks)
    end
    function Chain(sequence...)
        blocknum = length(sequence)
        blocks = Vector(undef,blocknum)
        for i = 1:blocknum
            blocks[i] = sequence[i]
        end
        new(blocknum, blocks)
    end
end


function paramsof(c::Chain)
    params = Vector{Variable}(undef,0)
    for i = 1:c.blocknum
        p = paramsof(c.blocks[i])
        if p != nothing
            append!(params, p)
        end
    end
    return params
end


function nparamsof(c::Chain)
    k = 0
    for i = 1:c.blocknum
        k += nparamsof(c.blocks[i])
    end
    return k
end


function resethidden(c::Chain)
    for i = 1:c.blocknum
        if c.blocks[i] isa irnn
            resethidden(c.blocks[i])
        end
        if c.blocks[i] isa IRNN
            resethidden(c.blocks[i])
        end
        # if c.blocks[i] isa lstm
        #     resethidden(c.blocks[i])
        # end
        # if c.blocks[i] isa gru
        #     resethidden(c.blocks[i])
        # end
        # if c.blocks[i] isa LSTM
        #     resethidden(c.blocks[i])
        # end
        # if c.blocks[i] isa GRU
        #     resethidden(c.blocks[i])
        # end
    end
end


function forward(c::Chain, input::Variable)
    x = forward(c.blocks[1], input)
    for i = 2:c.blocknum
        x = forward(c.blocks[i], x)
    end
    return x
end


function predict(c::Chain, input)
    x = predict(c.blocks[1], input)
    for i = 2:c.blocknum
        x = predict(c.blocks[i], x)
    end
    return x
end
