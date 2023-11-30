from memory.unsafe import Pointer
from random import random_float64
from python import Python

from .engine import Value


@always_inline
fn one_hot(vec: DynamicVector[Pointer[String]]) -> DynamicVector[Pointer[Value]]:
    pass


@always_inline
fn reverse(inout vec: DynamicVector[Pointer[Value]]) -> DynamicVector[Pointer[Value]]:
    var reversed: DynamicVector[Pointer[Value]] = DynamicVector[Pointer[Value]](
        len(vec)
    )
    for i in range(len(vec) - 1, -1, -1):
        reversed.push_back(vec[i])
    return reversed


@always_inline
fn to_float32(v: Float64) -> Float32:
    return v.cast[DType.float32]()


@always_inline
fn make_moons(
    n_samples: Int, noise: Float64
) raises -> (Pointer[Pointer[Pointer[Value]]], Pointer[Pointer[Value]]):
    let sklearn = Python.import_module("sklearn.datasets")
    let numpy = Python.import_module("numpy")
    let out = sklearn.make_moons(n_samples)
    let py_X = out[0]
    let py_y = out[1]

    let X = Pointer[Pointer[Pointer[Value]]].alloc(n_samples)
    for i in range(n_samples):
        let row = Pointer[Pointer[Value]].alloc(2)
        for j in range(2):
            let v_f64: Float64 = py_X[i][j].to_float64()
            let v_f32: Float32 = to_float32(v_f64)
            # Add some noise
            let noise_v: Float32 = to_float32(random_float64(-noise, noise))
            let value = Value(v_f32 + noise_v)
            let ptr_value = Pointer[Value].alloc(1)
            ptr_value.store(value)
            row.store(j, ptr_value)
        X.store(i, row)

    let y = Pointer[Pointer[Value]].alloc(n_samples)
    for i in range(n_samples):
        let v_f64: Float64 = py_y[i].to_float64()
        let v_f32: Float32 = to_float32(v_f64)
        # Make y be -1 or 1
        let value = Value(v_f32 * 2 - 1)
        let ptr_value = Pointer[Value].alloc(1)
        ptr_value.store(value)
        y.store(i, ptr_value)

    return X, y


@always_inline
fn print_datasets(
    X: Pointer[Pointer[Pointer[Value]]], y: Pointer[Pointer[Value]], n_samples: Int
):
    print("X:")
    for i in range(n_samples):
        let row = X.load(i)
        print(i, row.load(0).load().data.load(), row.load(1).load().data.load())

    print("y:")
    for i in range(n_samples):
        print(i, y.load(i).load().data.load())


@always_inline
fn plot(
    X: Pointer[Pointer[Pointer[Value]]],
    y: Pointer[Pointer[Value]],
    n_samples: Int,
    filename: String,
    title: String,
) raises:
    let np = Python.import_module("numpy")
    let plt = Python.import_module("matplotlib.pyplot")
    let x0 = np.zeros(n_samples, np.float32)
    let x1 = np.zeros(n_samples, np.float32)
    let yy = np.zeros(n_samples, np.float32)

    for i in range(n_samples):
        _ = x0.itemset(i, X.load(i).load(0).load().data.load())
        _ = x1.itemset(i, X.load(i).load(1).load().data.load())
        _ = yy.itemset(i, y.load(i).load().data.load() + 100)

    _ = plt.title(title)
    _ = plt.scatter(x0, x1, 10, yy)
    _ = plt.savefig(filename)
