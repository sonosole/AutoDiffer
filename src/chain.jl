mutable struct Chain
    blocks::Array{Block,1}
    function Chain(sequence::Array{Block,1})
        len = length(sequence)
        blocks = Vector(undef,len)
        for i = 1:len
            blocks[i] = sequence[i]
        end
    end
end


function forward(seq::Chain, input::Variable)
    var = Vector(undef, 0)
    x,w = forward(seq.blocks[1], input)
    push!(var, w)

    for block in seq.blocks[2:end]
        x, w = forward(block, x)
        push!(var, w)
    end
    return  x, var
end


#
