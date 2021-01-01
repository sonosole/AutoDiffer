function testctcfn()
    S = 5
    T = 10000
    x = 0.000005*rand(S,T)
    P = Variable(x,true)

    r,loglikely = DNN_CTC_Without_Softmax(P, [2 3 4 5 2 3 4 5])
    println("loglikely = ",loglikely)
    backward()
    return nothing
end
