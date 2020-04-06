function LogSum2Exp(a::Real, b::Real)
	if a <= -1.797e308
        a = -1.797e308
    end
	if b <= -1.797e308
        b = -1.797e308
    end
	return (max(a,b) + log(1.0 + exp(-abs(a-b))));
end


function LogSum3Exp(a::Real, b::Real, c::Real)
    return LogSum2Exp(LogSum2Exp(a,b),c)
end


function LogSumExp(a)
    tmp = -1.797e308
    for i = 1:length(a)
        tmp = LogSum2Exp(tmp,a[i])
    end
    return tmp
end


function CTC(p::Array{TYPE,2}, seq) where TYPE
    # inputs:
    # p      -- probability of softmax output
    # seq    -- label seq
    # outputs:
    # r      -- target of softmax output
    # logsum -- log-likelyhood

    S,T = size(p)
    L = length(seq)*2 + 1
    a = fill(-floatmax(), L,T) # alpha
    b = fill(-floatmax(), L,T) # beta
    r = fill(-floatmax(), S,T) # gamma

    if L>1
        a[1,1] = log(p[    1, 1])
        a[2,1] = log(p[seq[1],1])
        b[L-1,T] = 0.0
        b[L-0,T] = 0.0
    else
        a[1,1] = log(p[1,1])
        b[L,T] = 0.0
    end

    # --- forward ---
    for t = 2:T
        first = max(1,L-2*(T-t)-1);
        lasst = min(2*t,L);
        for s = first:lasst
            i = div(s,2);
            if s==1
                a[s,t] = a[s,t-1] + log(p[1,t])
            elseif mod(s,2)==1
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[1,t])
            elseif s==2
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[seq[i],t])
            elseif seq[i]==seq[i-1]
				a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[seq[i],t])
            else
                a[s,t] = LogSum3Exp(a[s,t-1], a[s-1,t-1], a[s-2,t-1]) + log(p[seq[i],t])
            end
        end
    end

    # --- backward ---
    for t = T-1:-1:1
        first = max(1,L-2*(T-t)-1)
        lasst = min(2*t,L)
        for s = first:lasst
            i = div(s,2)
            j = div(s+1,2)
            if s==L
                b[s,t] = b[s,t+1] + log(p[1,t+1])
            elseif mod(s,2)==1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[1,t+1]), b[s+1,t+1] + log(p[seq[j],t+1]))
            elseif s==L-1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[1,t+1]))
            elseif seq[i]==seq[i+1]
				b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[1,t+1]))
            else
                b[s,t] = LogSum3Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[1,t+1]), b[s+2,t+1] + log(p[seq[i+1],t+1]))
            end
        end
    end

    logsum = - floatmax()
    for s = 1:L
        logsum = LogSum2Exp(logsum, a[s,1] + b[s,1])
    end

    for s = 1:L
        i = div(s,2)
        if mod(s,2)==1
            r[1,:] .= LogSum2Exp.(r[1,:], a[s,:] + b[s,:])
        else
            r[seq[i],:] .= LogSum2Exp.(r[seq[i],:], a[s,:] + b[s,:])
        end
    end
    r .= exp.(r .- logsum)
    return r, logsum
end


function testctc()
    S = 5
    T = 30000
    x = 0.01*rand(S,T)
    P = x

    for t=1:T
        P[:,t] = exp.( x[:,t] .- maximum(x[:,t]) )
        P[:,t] = P[:,t] ./ sum(P[:,t])
    end
    tic = time()
    r,loglikely = CTC(P,[2 3 4 5 2 3 4 5])
    toc = time()
    println("loglikely = ",loglikely)
    println("ctc_time  = ",(toc-tic)*1000," ms")
    Gadfly.plot(
    layer(y=r[1,:],Geom.line,Theme(default_color=colorant"red")),
    layer(y=r[2,:],Geom.line,Theme(default_color=colorant"yellow")),
    layer(y=r[3,:],Geom.line,Theme(default_color=colorant"blue")),
    layer(y=r[4,:],Geom.line,Theme(default_color=colorant"green")),
    layer(y=r[5,:],Geom.line,Theme(default_color=colorant"orange"))
    )


end


@time testctc()

function DNN_CTC_Without_Softmax(var::Variable, seq)
    row, col = size(var.value)
    fac = length(seq) / col
    out = Variable(zeros(row, col), var.trainable)

    Xmax = maximum(var.value, dims=1)
    out.value = exp.(var.value .- Xmax)
    out.value ./= sum(out.value, dims=1)
    r, loglikely = CTC(out.value, seq)

    if var.trainable
        function DNN_CTC_Without_Softmax_Backward()
            var.delta += (out.value - r) .* fac
        end
        push!(graph.backward, DNN_CTC_Without_Softmax_Backward)
    end
    return out, loglikely
end


function DNN_CTC_Without_Softmax(x::Array)
    xmax = maximum(x, dims=1)
    prob = exp.(x .- xmax)
    psum = sum(prob, dims=1)
    return (prob ./ psum)
end
