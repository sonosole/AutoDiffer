NINF = -floatmax(Float32)


function LogSum2Exp(a::Real, b::Real)
	if a <= NINF
        a = NINF
    end
	if b <= NINF
        b = NINF
    end
	return (max(a,b) + log(1.0 + exp(-abs(a-b))));
end


function LogSum3Exp(a::Real, b::Real, c::Real)
    return LogSum2Exp(LogSum2Exp(a,b),c)
end


function LogSumExp(a)
    tmp = NINF
    for i = 1:length(a)
        tmp = LogSum2Exp(tmp,a[i])
    end
    return tmp
end


function CTC(p::Array{TYPE,2}, seq) where TYPE
    # inputs:
    # p      -- probability of softmax output
    # seq    -- label seq like [9 2 3 6 5 2 2 3]
	#           1 is blank,so minimum of it is 2
    # outputs:
    # r      -- target of softmax's output
    # logsum -- log-likelyhood

    S,T = size(p)
    L = length(seq)*2 + 1
    a = fill(NINF, L,T) # alpha = p(s[k,t], x[1:t])
    b = fill(NINF, L,T) # beta  = p(x[t+1:T] | s[k,t])
    r = fill(NINF, S,T) # gamma = classWiseSum(alpha * beta)

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

    logsum = NINF
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
    return r, -logsum
end


function greedysearch(x)
	# 除了blank作为吸收态外，还有一个单独的状态作为集外词的吸收态
	# 并且约定将 blank 映射到 1，OOV映射到 2,集内词从3开始映射.
    dog =  1
    out = [1]
    idx = argmax(x,dims=1)
    for i = 1:length(idx)
        maxid = idx[i][1]
        if (maxid != 1) && (maxid != 2) && (maxid != out[dog])
            push!(out,maxid)
            dog += 1
        end
    end
    return out[2:end]
end


function testctc()
    S = 500
    T = 100
    x = 0.01*rand(S,T)
    P = x

    for t=1:T
        P[:,t] = exp.( x[:,t] .- maximum(x[:,t]) )
        P[:,t] = P[:,t] ./ sum(P[:,t])
    end
    tic = time()
    r,loglikely = CTC(P,[3 3 4 5 2 3 4 5])
    toc = time()
    println("loglikely = ",loglikely/T," (timeSteps averaged)")
    println("ctc_time  = ",(toc-tic)*1000," ms")
	# using Gadfly
    # Gadfly.plot(
    # layer(y=r[1,:],Geom.line,Theme(default_color=colorant"red")),
    # layer(y=r[2,:],Geom.line,Theme(default_color=colorant"yellow")),
    # layer(y=r[3,:],Geom.line,Theme(default_color=colorant"blue")),
    # layer(y=r[4,:],Geom.line,Theme(default_color=colorant"green")),
    # layer(y=r[5,:],Geom.line,Theme(default_color=colorant"orange"))
    # )
end


function DNN_CTC_With_Softmax(var::Variable, seq)
    row, col = size(var)
    fac = (length(seq)+1) / col
    out = Variable(zeros(row, col), var.trainable)
	out.value = softmax(var.value)
    r, loglikely = CTC(out.value, seq)

    if var.trainable
        function DNN_CTC_With_Softmax_Backward()
            var.delta += (out.value - r) .* fac
        end
        push!(graph.backward, DNN_CTC_With_Softmax_Backward)
    end
    return out, loglikely
end


function indexbounds(sizeArray)
	# assert sizeArray has no 0 element
    acc = 0
    num = length(sizeArray)
    s = ones(Int,num,1)
    e = ones(Int,num,1)
    for i = 1:num
        s[i] += acc
        e[i] = s[i] + sizeArray[i] - 1
        acc += sizeArray[i]
    end
    return (s,e)
end


function DNN_Batch_CTC_With_Softmax(var::Variable, seq, inputSizeArray, labelSizeArray)
    row, col = size(var)
    out = Variable(zeros(row, col), var.trainable)

	Loglikely = 0.0
	batchsize = length(inputSizeArray)
	out.value = softmax(var.value)
	gamma     = zero(out.value)
	sidI,eidI = indexbounds(inputSizeArray)
	sidL,eidL = indexbounds(labelSizeArray)

	for i = 1:batchsize
		IDI = sidI[i]:eidI[i]
		IDL = sidL[i]:eidL[i]
		len = length(IDI)
		fac = (length(IDL)+1) / len
		gamma[:,IDI], loglikely = CTC(out.value[:,IDI], seq[IDL])
		gamma[:,IDI] .*= fac
		out.value[:,IDI] .*= fac
		Loglikely += loglikely
	end

    if var.trainable
        function DNN_Batch_CTC_With_Softmax_Backward()
            var.delta += out.value - gamma
        end
        push!(graph.backward, DNN_Batch_CTC_With_Softmax_Backward)
    end
    return out, Loglikely/batchsize
end
