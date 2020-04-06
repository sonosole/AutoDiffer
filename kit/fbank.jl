
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
