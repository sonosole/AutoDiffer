using FFTW


mutable struct featparams
    # 分帧参数
    winLength::Int
    winShift::Int
    # 梅尔滤波参数
    numBanks::Int
    numFFT::Int
    alpha::Real
    fs::Int
    # 特征拼接参数
    leftpad::Int
    rightpad::Int
    stackShift::Int
    # 自动计算参数
    maxfreq::Int     # 频域最大有效频率下标
    epsilon::Real    # 防止下溢
    stackSize::Int   # 超帧含有的小帧个数
    featLength::Int  # 超帧维度=拼帧数*每帧特征维度
    featShift::Int   # 超帧位移=移帧数*每帧特征维度

    function featparams(;
        winLength  = 256,
        winShift   = 128,
        numBanks   = 32,
        numFFT     = 256,
        alpha      = 0.97,
        fs         = 16000,
        leftpad    = 4,
        rightpad   = 3,
        stackShift = 3,
        epsilon    = 1e-9)
        maxfreq    = numFFT>>1
        stackSize  = leftpad + 1 + rightpad   # 超帧含有的小帧个数
        featLength = stackSize * numBanks     # 超帧维度=拼帧数*每帧特征维度
        featShift  = stackShift * numBanks    # 超帧位移=移帧数*每帧特征维度
        new(winLength,winShift,numBanks,numFFT,alpha,fs,leftpad,rightpad,
            stackShift,maxfreq,epsilon,stackSize,featLength,featShift)
    end
end


function filterbanks(numBanks::Int, numFFT::Int, fs::Int)
    # numBanks - 滤波带个数,比如 32 assert(numBanks<numFFT/2)
    # numFFT   - FFT点数,比如 256,128 etc.
    # fs       - 采样率,比如 16000kHz/8000kHz
    MAX   = numFFT>>1;                    # 正频率部分的下标最大值
    freq  = (0:(MAX-1))/MAX * fs/2;       # 下标映射到频率
    Fmel  = 2595*log10.(1 .+ freq/700);   # 频率映射到梅尔频率
    dFmel = Fmel[MAX]/(numBanks+1);       # 将Mel带平分成
    bank  = zeros(numBanks, MAX);         # 滤波器的频域权重系数
    cFmel = 0.0;                          # 每个Mel频带的中心Mel频率
    for n = 1:numBanks
        cFmel = cFmel + dFmel
        for m = 1:MAX
            if ( Fmel[m] >= cFmel-dFmel ) && ( Fmel[m] <= cFmel+dFmel )
                bank[n,m] = 1.0 - abs( Fmel[m] - cFmel )/dFmel
            else
                bank[n,m] = 0.0
            end
        end
    end
    return bank
end


function window(winLen::Int)
	hamming = 0.54 .- 0.46 .* cos.( 2*pi .* (0:(winLen-1))/(winLen-1) )
end


function filterwav(data, alpha)
    return (data[2:end] - alpha .* data[1:end-1])
end


function splitwav(data, win, winLength::Int, winShift::Int)
    numFrame = floor(Int, (length(data)-winLength)/winShift) + 1
    firstIds = (0:(numFrame-1)) .* winShift .+ 1     # 帧起始下标
    lasstIds = firstIds .+ (winLength - 1)           # 帧结束下标
    frames   = zeros(winLength, numFrame)
    for i = 1:numFrame
        frames[:,i] = data[firstIds[i]:lasstIds[i]] .* win
    end
    return frames, numFrame
end


function initfeat(params::featparams)
    winLength  = params.winLength
    winShift   = params.winShift
    numBanks   = params.numBanks
    numFFT     = params.numFFT
    maxfreq    = params.maxfreq
    alpha      = params.alpha
    fs         = params.fs
    stackSize  = params.stackSize
    stackShift = params.stackShift
    featLength = params.featLength
    featShift  = params.featShift
    epsilon    = params.epsilon
    winfunc    = window(winLength)
    melbank    = filterbanks(numBanks, numFFT, fs)

    function offlinefeat(wav)
        # 滤波 + 分帧 + 提取小特征 + 拼接小特征
        wavedata  = filterwav(wav, alpha)
        frames, n = splitwav(wavedata, winfunc, winLength, winShift)
        numsfeats = floor(Int, (n - stackSize)/stackShift) + 1  # 超帧数
        superfeat = zeros(featLength, numsfeats)                # 超帧数组

        tmp = fft(frames,1)                                 # 时域到频域,按列计算
        pow = abs2.(tmp[1:maxfreq,:])                       # 功率谱,提取有用部分
        mel = log.(melbank * pow .+ epsilon)                # 对数梅尔功率谱

        firstIds = (0:numsfeats-1) .* featShift .+ 1        # 超帧起始下标数组
        lasstIds = firstIds .+ (featLength - 1)             # 超帧结束下标数组
        for i = 1:numsfeats
            superfeat[:,i] = mel[firstIds[i]:lasstIds[i]]
        end
        return superfeat
    end
    return offlinefeat
end


function show(f::featparams)
    println("--------------------")
    println(" winLength  = $(f.winLength)")
    println(" winShift   = $(f.winShift)")
    println(" numBanks   = $(f.numBanks)")
    println(" numFFT     = $(f.numFFT)")
    println(" alpha      = $(f.alpha)")
    println(" fs         = $(f.fs)")
    println(" leftpad    = $(f.leftpad)")
    println(" rightpad   = $(f.rightpad)")
    println(" stackShift = $(f.stackShift)")
    println(" maxfreq    = $(f.maxfreq)")
    println(" epsilon    = $(f.epsilon)")
    println(" stackSize  = $(f.stackSize)")
    println(" featLength = $(f.featLength)")
    println(" featShift  = $(f.featShift)")
    println("--------------------")
end


# function testfeat()
#     p = featparams()
#     getfeat = initfeat(p)
#     wav = chirp(1,16000,0.0,8000.0)
#     tic = time()
#     feat = getfeat(wav)
#     toc = time()
#     ms = (toc-tic)*1000
#     rt = floor(1000 / ms)
#     println("=============================================================")
#     println("time to extrac features from 1 sec wav: ",ms," ms")
#     println(" rts to extrac features from 1 sec wav: ", rt)
#     println("=============================================================")
#     return feat
# end


# 实时速度约 895 倍 (i7-3630QM CPU@2.4GHz)
# 相当于 1 秒可以处理 895 秒的16KHz@16bit的 wav 数据 @ :
# winLength  = 256,
# winShift   = 128,
# numBanks   = 32,
# numFFT     = 256,
# alpha      = 0.9,
# fs         = 16000,
# leftpad    = 4,
# rightpad   = 3,
# stackShift = 3,
# epsilon    = 1e-9
