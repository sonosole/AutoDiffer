# -- 模型单元如 dense,rnn ...
abstract type Block end

# -- 节点类型 --
abstract type AbstractNode end


mutable struct Variable{T} <: AbstractNode
    value::T # 计算值
    delta::T # 梯度值
    trainable::Bool  # 是否参与训练
    parents::Vector  # 父节点列表
    children::Vector # 子节点列表
    function Variable(var::T) where T
        new{T}(var, zero(var), false, Vector(undef,0), Vector(undef,0))
    end

    function Variable(var::T, trainable::Bool) where T
        new{T}(var, zero(var), trainable, Vector(undef,0), Vector(undef,0))
    end
end


mutable struct Graph
    backward::Vector  # 存储反向传播的操作
    operator::Vector  # 操作类型
    backProp::Bool    # 是否需要反向传播
    function Graph()
        new(Vector(undef,0), Vector(undef,0), true)
    end
    function Graph(NeedBackProp::Bool)
        new(Vector(undef,0), Vector(undef,0), NeedBackProp)
    end
end
