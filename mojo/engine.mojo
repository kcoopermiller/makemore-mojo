from memory import memset_zero
from memory.unsafe import Pointer
from random import random_float64

from .utils import to_float32, reverse


@register_passable("trivial")
struct Value:
    var data: Pointer[Float32]
    var grad: Pointer[Float32]
    var l: Pointer[Int]
    var r: Pointer[Int]
    var _op: StringRef

    fn __init__(data: Float32) -> Self:
        let ptr_data: Pointer[Float32] = Pointer[Float32].alloc(1)
        ptr_data.store(data)

        let ptr_grad: Pointer[Float32] = Pointer[Float32].alloc(1)
        ptr_grad.store(0.0)

        return Self {
            data: ptr_data,
            grad: ptr_grad,
            l: Pointer[Int].get_null(),
            r: Pointer[Int].get_null(),
            _op: "",
        }

    # Forward pass

    @always_inline
    fn __add__(self, inout other: Pointer[Value]) -> Value:
        var new_value: Value = Value(0)

        new_value.data.store(self.data.load() + other.load().data.load())

        let ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        new_value.l = ptr_l.bitcast[Int]()
        new_value.r = other.bitcast[Int]()

        new_value._op = "+"

        return new_value

    @always_inline
    fn __add__(self, inout other: Float32) -> Value:
        let new_value: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(new_value)
        return self.__add__(ptr_v)

    @always_inline
    fn __add__(self, inout other: Value) -> Value:
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)
        return self + ptr_v

    @always_inline
    fn __mul__(self, inout other: Pointer[Value]) -> Value:
        var new_value: Value = Value(0)

        new_value.data.store(self.data.load() * other.load().data.load())

        let ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        new_value.l = ptr_l.bitcast[Int]()
        new_value.r = other.bitcast[Int]()

        new_value._op = "*"

        return new_value

    @always_inline
    fn __mul__(self, inout other: Float32) -> Value:
        let new_value: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(new_value)
        return self.__mul__(ptr_v)

    @always_inline
    fn __mul__(self, inout other: Value) -> Value:
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)
        return self * ptr_v

    @always_inline
    fn __pow__(self, inout other: Pointer[Value]) -> Value:
        var new_value: Value = Value(0)

        # Using other only as a holder for data, it doens't backprop (we don't backward(r) bellow)
        # and we leave new_value.r as null pointer here
        new_value.data.store(self.data.load() ** other.load().data.load())

        let ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        new_value.l = ptr_l.bitcast[Int]()
        new_value.r = other.bitcast[Int]()
        # Only has one child
        new_value._op = "**"

        return new_value

    @always_inline
    fn __pow__(self, inout other: Value) -> Value:
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)
        return self**ptr_v

    @always_inline
    fn __pow__(self, inout other: Float32) -> Value:
        let new_value: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(new_value)
        return self.__pow__(ptr_v)

    @always_inline
    fn relu(self) -> Value:
        var new_value: Value = Value(0)

        var data: Float32 = 0
        if self.data.load() >= 0:
            data = self.data.load()
        new_value.data.store(data)

        let ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        new_value.l = ptr_l.bitcast[Int]()
        # Only has one child
        new_value._op = "ReLU"

        return new_value

    @always_inline
    fn __neg__(self) -> Value:
        var data: Float32 = -1
        return self * data

    @always_inline
    fn __radd__(self, inout other: Value) -> Value:
        return self + other

    @always_inline
    fn __radd__(self, inout other: Float32) -> Value:
        return self + other

    @always_inline
    fn __sub__(self, inout other: Pointer[Value]) -> Value:
        let value: Value = other.load()
        var data: Float32 = -1
        var mul: Value = value * data
        var ptr_mul: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_mul.store(mul)
        return self + ptr_mul

    @always_inline
    fn __sub__(self, inout other: Value) -> Value:
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)
        return self - ptr_v

    @always_inline
    fn __sub__(self, inout other: Float32) -> Value:
        let new_value: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(new_value)
        return self.__sub__(ptr_v)

    @always_inline
    fn __rsub__(self, inout other: Pointer[Value]) -> Value:
        return self - other

    fn __rsub__(self, inout other: Float32) -> Value:
        return self - other

    @always_inline
    fn __rmul__(self, inout other: Pointer[Value]) -> Value:
        return self * other

    @always_inline
    fn __rmul__(self, inout other: Float32) -> Value:
        return self * other

    @always_inline
    fn __truediv__(self, inout other: Value) -> Value:
        var powie: Float32 = -1.0
        var pow: Value = other**powie
        return self * pow

    @always_inline
    fn __rtruediv__(self, inout other: Value) -> Value:
        var powie: Float32 = -1.0
        var pow: Value = self**powie
        return other * pow

    @always_inline
    fn __truediv__(self, inout other: Float32) -> Value:
        return self / other

    @always_inline
    fn __rtruediv__(self, inout other: Float32) -> Value:
        return other / self

    # Backward pass

    @staticmethod
    @always_inline
    fn _backward(inout v: Pointer[Value]):
        let op: String = v.load()._op
        if op == "":
            return
        if op == "+":
            Value.backward_add(v)
        elif op == "*":
            Value.backward_mul(v)
        elif op == "**":
            Value.backward_pow(v)
        elif op == "ReLU":
            Value.backward_relu(v)
        else:
            print("OP not supported", op)

    @staticmethod
    @always_inline
    fn backward_add(inout v: Pointer[Value]):
        let vv: Value = v.load()
        if vv.l == Pointer[Int].get_null():
            return
        let l: Value = vv.l.bitcast[Value]().load()
        l.grad.store(l.grad.load() + vv.grad.load())

        if vv.r == Pointer[Int].get_null():
            return
        let r: Value = vv.r.bitcast[Value]().load()
        r.grad.store(r.grad.load() + vv.grad.load())

    @staticmethod
    @always_inline
    fn backward_mul(inout v: Pointer[Value]):
        let vv: Value = v.load()
        if vv.l == Pointer[Int].get_null():
            return
        let l: Value = vv.l.bitcast[Value]().load()
        if vv.r == Pointer[Int].get_null():
            return
        let r = vv.r.bitcast[Value]().load()
        l.grad.store(l.grad.load() + (r.data.load() * vv.grad.load()))
        r.grad.store(r.grad.load() + (l.data.load() * vv.grad.load()))

    @staticmethod
    @always_inline
    fn backward_pow(inout v: Pointer[Value]):
        let vv: Value = v.load()
        if vv.l == Pointer[Int].get_null():
            return
        let l: Value = vv.l.bitcast[Value]().load()
        if v.load().r == Pointer[Int].get_null():
            return
        let r: Value = vv.r.bitcast[Value]().load()
        let other = r.data.load()
        l.grad.store(
            l.grad.load() + (other * l.data.load() ** (other - 1) * vv.grad.load())
        )

    @staticmethod
    @always_inline
    fn backward_relu(inout v: Pointer[Value]):
        let vv: Value = v.load()
        if vv.l == Pointer[Int].get_null():
            return
        let l: Value = vv.l.bitcast[Value]().load()
        # TODO Can be Int?
        var data: Float32 = 0
        if vv.data.load() > 0:
            data = 1
        let update: Float32 = data * v.load().grad.load()
        l.grad.store(l.grad.load() + update)

    @staticmethod
    fn build_topo(
        inout ptr_v: Pointer[Value],
        inout visited: DynamicVector[Pointer[Value]],
        inout topo: DynamicVector[Pointer[Value]],
    ):
        if ptr_v == Pointer[Value].get_null():
            return
        var is_visited: Bool = False
        let size: Int = len(visited)
        for i in range(size):
            if ptr_v == visited[i]:
                is_visited = True
        if not is_visited:
            visited.push_back(ptr_v)
            # Make sure we don't try to access null pointers (e.g. on pow
            # where we don't have the right child)
            if ptr_v.load().l != Pointer[Int].get_null():
                var ptr_l: Pointer[Value] = ptr_v.load().l.bitcast[Value]()
                if ptr_l != Pointer[Value].get_null():
                    Value.build_topo(ptr_l, visited, topo)
            if ptr_v.load().r != Pointer[Int].get_null():
                var ptr_r: Pointer[Value] = ptr_v.load().r.bitcast[Value]()
                if ptr_r != Pointer[Value].get_null():
                    Value.build_topo(ptr_r, visited, topo)
            topo.push_back(ptr_v)

    @always_inline
    fn backward(inout self):
        var visited: DynamicVector[Pointer[Value]] = DynamicVector[Pointer[Value]]()
        var topo: DynamicVector[Pointer[Value]] = DynamicVector[Pointer[Value]]()
        var ptr_self: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_self.store(self)
        Value.build_topo(ptr_self, visited, topo)
        self.grad.store(1.0)
        var reversed: DynamicVector[Pointer[Value]] = reverse(topo)
        for i in range(len(reversed)):
            Value._backward(reversed[i])
        visited.clear()
        topo.clear()
        reversed.clear()
        ptr_self.free()

    @always_inline
    fn print(inout self):
        print(
            "<Value",
            "data:",
            self.data.load(),
            "grad:",
            self.grad.load(),
            "op:",
            self._op,
            ">",
        )

    @always_inline
    fn print(inout self, label: StringRef):
        print(
            "<Value",
            "label:",
            label,
            "data:",
            self.data.load(),
            "grad:",
            self.grad.load(),
            "op:",
            self._op,
            ">",
        )
