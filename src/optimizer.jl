



```@docs
Nesterov
RMSProp
ADAM
AdaMax
ADAGrad
ADADelta
AMSGrad
NADAM
ADAMW
```

abstract type Optimizer end



mutable struct Descent <: Optimizer
    lr
    Descent() = new(7e-5)
    Descent(learningRate) = new(learningRate)
end


mutable struct Momentum <: Optimizer
  lr
  p
  function Momentum(learningRate)
      new(learningRate, 0.9)
  end
  function Momentum(learningRate, inertia)
      new(learningRate, inertia)
  end
end
