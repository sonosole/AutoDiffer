# -- 模型单元如 dense,rnn ...
abstract type Block end


# -- 节点类型 --
abstract type AbstractNode end


mutable struct Variable{T} <: AbstractNode
    value::T         # 计算值
    delta::T         # 梯度值
    trainable::Bool  # 是否参与训练
    function Variable(var::T) where T
        new{T}(var, zero(var), false)
    end

    function Variable(var::T, trainable::Bool) where T
        new{T}(var, zero(var), trainable)
    end
end


mutable struct Graph
    # 存储反向传播操作
    backward::Vector
    function Graph()
        new(Vector(undef,0))
    end
end


# -- 全局图变量，存储所有反向运算的中间
# -- 变量，但是在参数更新后需要将其置空
global graph = Graph()
