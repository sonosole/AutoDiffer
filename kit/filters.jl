function LowPassFilter(fc, fs, N)
    @assert(fs>=fc);
    n = 1:N;
    w = 0.54 .- 0.46*cos.(6.283185307179586*(0:N-1)/(N-1));
    wc = fc / fs * 6.283185307179586;
    if mod(N,2)==1
        h = sin.( wc * (n .- 0.5*(N+1)) ) ./ (n .- 0.5*(N+1)) / pi;
        h[ (N+1)>>1 ] = wc/pi;
        h = h .* w;
    else
        h = sin.( wc * (n .- 0.5*(N+1)) ) ./ (n .- 0.5*(N+1)) / pi;
        h  = h .* w;
    end
    return h
end

function BandPassFilter(fL,fH,fs,N)
    @assert(fs >= fH >= fL);
    return LowPassFilter(fH, fs, N) - LowPassFilter(fL, fs, N);
end

function HighPassFilter(fc,fs,N)
    @assert(fs >= fc)
    return LowPassFilter(fs*0.5, fs, N) - LowPassFilter(fc, fs, N);
end


# using Plots
# default(show = true)
# plotly()
# plot(rand(4,4),linewidth=2,title="My Plot1")
