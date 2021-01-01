# Base.__precompile__(true)

# module AutoDiffer

include("./baseStruct.jl")
include("./baseOperator.jl")

include("./dropout.jl")

include("./linear.jl")
include("./dense.jl")
include("./rnn.jl")
include("./lstm.jl")
include("./indrnn.jl")
include("./indlstm.jl")
include("./rin.jl")

include("./conv1xd.jl")

include("./P1Relu.jl")
include("./maxout.jl")
include("./residual.jl")
include("./chain.jl")
include("./optimizer.jl")
include("./ctc.jl")

global RNNLIST = [rnn, rin, lstm, indrnn, indlstm, RNN, RIN, LSTM, INDLSTM];

# end  # moduleautodiffer
