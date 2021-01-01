abstract type Optimizer end


mutable struct Descent <: Optimizer
    lr::Union{Float32,Float64}
    decay::Union{Float32,Float64}
    name::String
    function Descent(;learnRate=1e-4,decay=1.0)
        new(learnRate, decay, "Descent")
    end
end


function update(d::Descent, params::Vector{Variable})
    lrate = d.lr
    d.lr *= d.decay
    update(params, lrate)
end


mutable struct Momentum <: Optimizer
    v::Vector
    lr::Union{Float32,Float64}
    p::Union{Float32,Float64}
    decay::Union{Float32,Float64}
    name::String
    function Momentum(params::Vector{Variable}; learnRate=1e-4, inertia=0.9, decay=1.0)
        num = length(params)
        vel = Vector(undef,num)
        for i = 1:num
           vel[i] = zero(params[i].delta)
        end
        new(vel, learnRate, inertia, decay, "Momentum")
    end
end


function update(m::Momentum, params::Vector{Variable}; clipvalue=1e4)
    vel = m.v
    lr  = m.lr
    p   = m.p
    m.lr *= m.decay
    for i = 1:length(params)
        vel[i] .= p .* vel[i] + clip.(params[i].delta, clipvalue)
        params[i].value .-= lr .* vel[i]
    end
end


mutable struct Adam <: Optimizer
    w1::Vector
    w2::Vector
    lr::Union{Float32,Float64}
    b1::Union{Float32,Float64}
    b2::Union{Float32,Float64}
    ϵ::Union{Float32,Float64}
    t::UInt
    b1t::Union{Float32,Float64}
    b2t::Union{Float32,Float64}
    decay::Union{Float32,Float64}
    name::String
    function Adam(params::Vector{Variable}; learnRate=1e-4, b1=0.9, b2=0.996, epsilon=1e-8, decay=1.0)
        num = length(params)
        w1  = Vector(undef,num)
        w2  = Vector(undef,num)
        for i = 1:num
            w1[i] = zero(params[i].delta)
            w2[i] = zero(params[i].delta)
        end
        new(w1,w2,learnRate, b1, b2, epsilon, 0, 1.0, 1.0, decay, "Adam")
    end
end


function update(a::Adam, params::Vector{Variable}; clipvalue=1.0)
    w1 = a.w1
    w2 = a.w2
    lr = a.lr
    b1 = a.b1
    b2 = a.b2
    ϵ  = a.ϵ

    a.t   += 1
    a.b1t *= b1
    a.b2t *= b2
    a.lr  *= a.decay
    b1t = a.b1t
    b2t = a.b2t

    for i = 1:length(params)
        r = sqrt(1-b2t) / (1-b1t) * lr
        g = clip.(params[i].delta, clipvalue)
        w1[i] .= b1 .* w1[i] + (1-b1) .* g
        w2[i] .= b2 .* w2[i] + (1-b2) .* (g .* g)
        params[i].value .-= r .* w1[i] ./ sqrt.(w2[i] .+ ϵ)
    end
end


function decay(params::Vector{Variable}; ratio=0.999)
    for p in params
        p.value .*= ratio
    end
end


function normclip(gradient, clipvalue)
    L2NormVal  = sqrt( sum(gradient .^ 2) / length(gradient) )
    Normalizer = clipvalue / L2NormVal
    if L2NormVal > clipvalue
        gradient .*= Normalizer
    end
    return gradient
end
