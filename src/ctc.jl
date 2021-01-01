LogZero = -floatmax(Float64)


function LogSum2Exp(a::Real, b::Real)
	if a <= LogZero
        a = LogZero
    end
	if b <= LogZero
        b = LogZero
    end
	return (max(a,b) + log(1.0 + exp(-abs(a-b))));
end


function LogSum3Exp(a::Real, b::Real, c::Real)
    return LogSum2Exp(LogSum2Exp(a,b),c)
end


function LogSumExp(a)
    tmp = LogZero
    for i = 1:length(a)
        tmp = LogSum2Exp(tmp, a[i])
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
    a = fill(LogZero, L,T) # alpha = p(s[k,t], x[1:t])
    b = fill(LogZero, L,T) # beta  = p(x[t+1:T] | s[k,t])
    r = zeros(S,T)         # gamma = classWiseSum(alpha * beta)

    if L>1
        a[1,1] = log(p[    1, 1])
        a[2,1] = log(p[seq[1],1])
        b[L-1,T] = 0.0
        b[L-0,T] = 0.0
    else
        a[1,1] = log(p[1,1])
        b[L,T] = 0.0
    end

    # --- forward in log scale ---
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

    # --- backward in log scale ---
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

    logsum = LogZero
    for s = 1:L
        logsum = LogSum2Exp(logsum, a[s,1] + b[s,1])
    end

	g = exp.((a + b) .- logsum)
    for s = 1:L
        if mod(s,2)==1
			r[1,:] .+= g[s,:]
        else
			i = div(s,2)
			r[seq[i],:] .+= g[s,:]
        end
    end
    return r, -logsum
end


function CTCGreedySearch(x)
	# blank 映射到 1
    hyp = []
    idx = argmax(x,dims=1)
    for i = 1:length(idx)
        maxid = idx[i][1]
        if (i!=1 && idx[i][1]==idx[i-1][1]) || (idx[i][1]==1)
            continue
        else
            push!(hyp,idx[i][1])
        end
    end
    return hyp
end


function testctc()
    S = 10
    T = 10
    x = 0.01*rand(S,T)
    P = x

    for t=1:T
        P[:,t] = exp.( x[:,t] .- maximum(x[:,t]) )
        P[:,t] = P[:,t] ./ sum(P[:,t])
    end
    tic = time()
	r,loglikely = CTC(P,[3 4 5])
    toc = time()
    println("loglikely = ",loglikely/T," (timeSteps averaged)")
    println("ctc_time  = ",(toc-tic)*1000," ms")
	p = lineplot(vec(r[1,:]))
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


function RNN_Batch_CTC_With_Softmax(var::Variable, seqlabel, inputSizeArray, labelSizeArray)
    dims,timesteps,batchsize = size(var)
    out = Variable(zeros(dims,timesteps,batchsize), var.trainable)
 	gamma = zeros(dims,timesteps,batchsize)
    LOGSCORE = 0.0

	for b = 1:batchsize
		t = 1:inputSizeArray[b]
	    Q = (labelSizeArray[b]+1) / inputSizeArray[b]
	    out.value[:,t,b] = softmax(var.value[:,t,b])
		gamma[:,t,b], likelyhood = CTC(out.value[:,t,b], seqlabel[b])
		gamma[:,t,b] .*= Q
		out.value[:,t,b] .*= Q
		# gamma[1,t,b] .*= 1.0/labelSizeArray[b]
		# out.value[1,t,b] .*= 1.0/labelSizeArray[b]
		LOGSCORE += likelyhood
	end

    if var.trainable
        function RNN_Batch_CTC_With_Softmax_Backward()
            var.delta += out.value - gamma
        end
        push!(graph.backward, RNN_Batch_CTC_With_Softmax_Backward)
    end
    return out, LOGSCORE/batchsize
end
