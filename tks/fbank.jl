using FFTW

function filterbanks(numBanks::UInt8, numFFT::UInt16, fs::UInt16)
    # numBanks - 滤波带个数,比如 32/24 assert(numBanks<numFFT/2)
    # Nfft     - FFT点数,比如 256/128 etc.
    # fs       - 采样率,比如 16000kHz/8000kHz
    MAX   = UInt16(floor(numFFT/2));         # 正频率部分的下标最大值
    freq  = (0:(MAX-1))/MAX * fs/2;          # 下标映射到频率
    Fmel  = 2595*log10.(1 .+ freq/700);      # 频率映射到梅尔频率
    dFmel = Fmel[MAX]/(numBanks+1);          # 将Mel带平分成 N+1 份
    bank  = zeros(numBanks, MAX);            # N个滤波器的频域权重系数
    cFmel = 0.0;                             # 每个Mel频带的中心Mel频率
    for n = 1:numBanks
        cFmel = cFmel + dFmel;
        for m = 1:MAX
            if ( Fmel[m] >= cFmel-dFmel ) && ( Fmel[m] <= cFmel+dFmel )
                bank[n,m] = 1 - abs( Fmel[m] - cFmel )/dFmel;
            else
                bank[n,m] = 0;
            end
        end
    end

    return bank

end

function window(winLen)
	hamming = 0.54 .- 0.46 .* cos.( 2*pi .* (0:(winLen-1))/(winLen-1) );
end

function filterwav(data, alpha)
    return (data[2:end] - alpha .* data[1:end-1])
end

function splitwav(data, winLength, winShift)
    numFrame = Int( floor((length(data)-winLength)/winShift) + 1 );
    firstIds = (0:(numFrame-1)) .* winShift .+ 1;     # 帧起始下标
    lasstIds = firstIds .+ (winLength - 1);           # 帧结束下标
    frames   = zeros(winLength, numFrame)
    for i = 1:numFrame
        frames[:,i] = data[firstIds[i]:lasstIds[i]];
    end
    return frames, numFrame
end


# function initFFT(numFFT)
#     if (numFFT & (numFFT-1))==0
#         id =  0:numFFT-1
#         wr =  cos.(2*pi/numFFT*id)
#         wi = -sin.(2*pi/numFFT*id)
#         println(id)
#     end
#     j = 0;
#     for i = 0:(numFFT-2)
#         println(i)
#         # if i < j
#         #     tmp   = id[j+1]
#         #     id[j+1] = id[i+1]
#         #     id[i+1] = tmp
#         # end
#         # k = numFFT>>1
#         # # while k<=j
#         # #     j -= k
#         # #     k /= 2
#         # # end
#         # j += k
#     end
#     return wr,wi,id
# end

# function fftorder(numFFT)
#     n = length(numFFT)
#     if (n & (n-1))==0
#         if n==1
#             return numFFT
#         else
#             return  [fftorder(numFFT[1:2:n-1]) fftorder(numFFT[2:2:n])]
#         end
#     else
#         println("input should have a length of 2^n")
#     end
# end

function testFBANK()
    numFFT::UInt16  = 256
    numBanks::UInt8 = 32
    fs::UInt16      = 16000
    alpha    = 0.97
    winLength = 256
    winShift  = 128
    banks = filterbanks(numBanks, numFFT, fs)
    wav = randn(16000,1)
    data = filterwav(wav, alpha)
    (frame, n) = splitwav(data, winLength, winShift)
    for i=1:n
        fft(frame[:,i]);
    end
end
