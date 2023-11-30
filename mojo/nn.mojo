from memory.unsafe import Pointer
from random import random_float64

from .engine import Value
from .utils import to_float32


@register_passable
struct Neuron:
    var w: Pointer[Pointer[Value]]
    var b: Pointer[Value]
    var nin: Int
    var nonlin: Bool

    fn __init__(nin: Int, nonlin: Bool = True) -> Self:
        let ptr_w = Pointer[Pointer[Value]].alloc(nin)
        for i in range(nin):
            let r = to_float32(random_float64(-1, 1))
            let v = Value(r)
            let ptr_v = Pointer[Value].alloc(1)
            ptr_v.store(v)
            ptr_w.store(i, ptr_v)

        let ptr_b = Pointer[Value].alloc(1)
        ptr_b.store(Value(0.0))

        return Self {w: ptr_w, b: ptr_b, nin: nin, nonlin: nonlin}

    @always_inline
    fn forward(inout self, inout x: Pointer[Pointer[Value]]) -> Pointer[Value]:
        var tot = Value(0.0)
        for i in range(self.nin):
            let w: Value = self.w.load(i).load()
            var x: Pointer[Value] = x.load(i)
            let s: Value = w * x
            # var ptr_s = Pointer[Value].alloc(1)
            # ptr_s.store(s)
            tot = tot + s
        let act = tot + self.b
        let ptr_act = Pointer[Value].alloc(1)
        if self.nonlin:
            ptr_act.store(act.relu())
            return ptr_act
        ptr_act.store(act)
        return ptr_act

    # fn parameters(self) -> Pointer[Pointer[Value]]:
    #     var params = Pointer[Pointer[Value]].alloc(self.nin + 1)
    #     for i in range(self.nin):
    #         params.store(i, self.w.load(i))
    #     params.store(self.nin, self.b)
    #     return params

    @always_inline
    fn parameters(self) -> DynamicVector[Pointer[Value]]:
        var params = DynamicVector[Pointer[Value]]()
        for i in range(self.nin):
            params.push_back(self.w.load(i))
        params.push_back(self.b)
        return params


@register_passable
struct Layer:
    var neurons: Pointer[Pointer[Neuron]]
    var nin: Int
    var nout: Int

    fn __init__(nin: Int, nout: Int, nonlin: Bool = True) -> Self:
        let neurons = Pointer[Pointer[Neuron]]().alloc(nout)
        for i in range(nout):
            let neuron = Neuron(nin, nonlin=nonlin)
            let ptr_neuron = Pointer[Neuron].alloc(1)
            ptr_neuron.store(neuron)
            neurons.store(i, ptr_neuron)
        return Self {neurons: neurons, nin: nin, nout: nout}

    @always_inline
    fn forward(inout self, inout x: Pointer[Pointer[Value]]) -> Pointer[Pointer[Value]]:
        let out = Pointer[Pointer[Value]].alloc(self.nout)
        for i in range(self.nout):
            let ptr_neuron: Pointer[Neuron] = self.neurons.load(i)
            var neuron = ptr_neuron.load()
            let ptr_neuron_out = neuron.forward(x)
            out.store(i, ptr_neuron_out)
        return out

    # fn parameters(self) -> Pointer[Pointer[Value]]:
    #     # For all neurons, get all their parameters
    #     var params = Pointer[Pointer[Value]].alloc(self.nout * self.nin)
    #     for i in range(self.nout):
    #         var params_neuron = self.neurons.load(i).load().parameters()
    #         for j in range(self.nin):
    #             params.store(i*self.nin + j, params_neuron.load(j))
    #     return params

    @always_inline
    fn parameters(self) -> DynamicVector[Pointer[Value]]:
        var params = DynamicVector[Pointer[Value]]()
        for i in range(self.nout):
            let params_neuron = self.neurons.load(i).load().parameters()
            for j in range(len(params_neuron)):
                params.push_back(params_neuron[j])
        return params


@register_passable
struct MLP:
    var layers: Pointer[Pointer[Layer]]
    var nlayers: Int

    fn __init__(nin: Int, nouts: DynamicVector[Int]) -> Self:
        let nnodes = len(nouts) + 1
        let nlayers = nnodes - 1
        let layers = Pointer[Pointer[Layer]]().alloc(nlayers)
        let layer = Layer(nin, nouts[0], nonlin=True)
        let ptr_layer = Pointer[Layer].alloc(1)
        ptr_layer.store(layer)
        layers.store(0, ptr_layer)

        for i in range(nlayers - 1):
            # Last layer is linear, others are nonlinear
            let nonlin = i != nlayers - 2
            let layer = Layer(nouts[i], nouts[i + 1], nonlin=nonlin)
            let ptr_layer = Pointer[Layer].alloc(1)
            ptr_layer.store(layer)
            layers.store(i + 1, ptr_layer)

        return Self {layers: layers, nlayers: nlayers}

    @always_inline
    fn forward(inout self, inout x: Pointer[Pointer[Value]]) -> Pointer[Pointer[Value]]:
        # var out: Pointer[Pointer[Value]] = x
        for i in range(self.nlayers):
            var layer: Layer = self.layers.load(i).load()
            # Not sure if this works
            x = layer.forward(x)
        return x

    # fn parameters(self) -> Pointer[Pointer[Value]]:
    #     # For all layers, get all their parameters
    #     var params = DynamicVector[Pointer[Value]]()
    #     for i in range(self.nlayers):
    #         var layer = self.layers.load(i).load()
    #         var layer_params = layer.parameters()
    #         for j in range(layer.nin * layer.nout):
    #             params.push_back(layer_params.load(j))
    #     # Hacky, return DynamicVector instead
    #     var ptr_params = Pointer[Pointer[Value]].alloc(len(params))
    #     for i in range(len(params)):
    #         ptr_params.store(i, params[i])
    #     return ptr_params

    @always_inline
    fn parameters(self) -> DynamicVector[Pointer[Value]]:
        # For all layers, get all their parameters
        var params = DynamicVector[Pointer[Value]]()
        for i in range(self.nlayers):
            let layer = self.layers.load(i).load()
            let layer_params = layer.parameters()
            for j in range(len(layer_params)):
                params.push_back(layer_params[j])
        return params

    @always_inline
    fn zero_grad(inout self):
        let params = self.parameters()
        for i in range(len(params)):
            params[i].load().grad.store(0, 0.0)
