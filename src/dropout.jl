mutable struct dropout <: Block
    p # dropout probibility
    dropout(   ) = new(0.1)
    dropout(pro) = new(pro)
end


function paramsof(m::dropout)
    return nothing
end


function forward(d::dropout, var::Variable)
    # 对网络激活节点进行灭活
    # 属于in-place操作,但是输入输出共享节点值引用
    row, col = size(var.value)
    RandMask = (rand(row, col) .< (1 - d.p)) ./ (1 - d.p)
    var.value .*= RandMask
    out = Variable(var.value, var.trainable)
    if var.trainable
        function dropoutBackward()
            var.delta += RandMask .* out.delta
        end
        push!(var.children, out)
        push!(out.parents,  var)
        push!(graph.backward, dropoutBackward)
    end
    return out
end


function predicate(d::dropout, input)
    return input
end
