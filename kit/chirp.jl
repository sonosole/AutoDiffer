
function chirp(T, fs, fl, fh)
    # T  -- 持续时间
    # fs -- 采样率
    # fl -- 开始频率
    # fh -- 结束频率
    fl = max(0,fl)
    fh = min(fs/2,fh)
    k  = (fh - fl)/T
    t  = 0:(fs * T - 1)
    y  = sin.(6.2831853*(k.*t .+ fl).*t)
end


function testchirp()
    return chirp(0.1,16000,0.0,8000.0)
end
