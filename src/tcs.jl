function TCS(p::Array{TYPE,2}, seq) where TYPE
	# BG + FG + Vocab 集合为 {BG,FG,Vocab...};
	# BG + FG + Vocab 映射为 [ 1, 2,    3...];
	S,T = size(p)
	L = length(seq)

	a = fill(LogZero, L,T) # alpha in log scale, p(s[k,t], x[1:t])
    b = fill(LogZero, L,T) # beta in log scale , p(x[t+1:T] | s[k,t])
    r = zeros(S,T)         # reduced gamma

	if L>1
		a[1,1] = log(p[    1, 1])
		a[2,1] = log(p[seq[1],1])
		b[L-1,T] = 0.0
		b[L-0,T] = 0.0
	else
		a[1,1] = log(p[1,1])
		b[L,T] = 0.0
	end

	for t = 2:T
	    for s = 1:L
	        if s!=1
				R = mod(s,3)
	            if R==1 || s==2 || R==0
	                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1])
	            elseif R==2
	                a[s,t] = LogSum3Exp(a[s,t-1], a[s-1,t-1], a[s-2,t-1])
	            end
	        else
	            a[s,t] = a[s,t-1]
	        end
	        a[s,t] += log(p[seq[s],t])
	    end
	end

	for t = T-1:-1:1
		for s = L:-1:1
			Q = b[s,t+1] + log(p[seq[s],t+1])
			if s!=L
				R = mod(s,3)
				V = b[s+1,t+1] + log(p[seq[s+1],t+1])
				if R==1 || R==2 || s==L-1
					b[s,t] = LogSum2Exp(Q, V)
				elseif R==0
					b[s,t] = LogSum3Exp(Q, V, b[s+2,t+1] + log(p[seq[s+2],t+1]))
				end
			else
				b[s,t] = Q
			end
		end
	end

	logsum = LogZero
    for s = 1:L
        logsum = LogSum2Exp(logsum, a[s,1] + b[s,1])
    end

	g = exp.((a + b) .- logsum)
    for s = 1:L
		r[seq[s],:] .+= g[s,:]
    end
    return r, -logsum
end


function RNN_Batch_TCS_With_Softmax(var::Variable, seqlabel, inputSizeArray, labelSizeArray)
    dims,timesteps,batchsize = size(var)
    out = Variable(zeros(dims,timesteps,batchsize), var.trainable)
 	gamma = zeros(dims,timesteps,batchsize)
    LOGSCORE = 0.0

	for b = 1:batchsize
		t = 1:inputSizeArray[b]
	    Q = (labelSizeArray[b]+1) / inputSizeArray[b]
	    out.value[:,t,b] = softmax(var.value[:,t,b])
		gamma[:,t,b], likelyhood = TCS(out.value[:,t,b], seqlabel[b])
		gamma[:,t,b] .*= Q
		out.value[:,t,b] .*= Q
		LOGSCORE += likelyhood
	end

    if var.trainable
        function RNN_Batch_TCS_With_Softmax_Backward()
            var.delta += out.value - gamma
        end
        push!(graph.backward, RNN_Batch_TCS_With_Softmax_Backward)
    end
    return out, LOGSCORE/batchsize
end


function TCSGreedySearch(x)
	# BG映射到 1
	# FG映射到 2
    hyp = []
    idx = argmax(x,dims=1)
    for i = 1:length(idx)
        maxid = idx[i][1]
        if (i!=1 && idx[i][1]==idx[i-1][1]) || (idx[i][1]==1) || (idx[i][1]==2)
            continue
        else
            push!(hyp,idx[i][1])
        end
    end
    return hyp
end
