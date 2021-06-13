# STensor, the _Stable_ Tensor class
STensor is an extension of the `Tensor` class used in the popular Pytorch deep learning library, which addresses overflow/underflow issues that occur when multiplicative operations are used frequently. For example, consider the following simple example of iterated matrix-vector multiplication:

```python
import torch

matrix = torch.eye(5) * 10
vector = torch.ones(5)

for i in range(100):
    vector = matrix @ vector

output = torch.log10(vector)
print(output)
# >>> tensor([nan, nan, nan, nan, nan])
```

It isn't hard to work out what the exact answer should be here, but overflow leads to a garbage output filled with `nan`s. This basic example is representative of a common problem that arises in hidden Markov models, probabilistic graphical models, tensor networks, and many other similar systems. Anyone implementing such models are typically expected to anticipate this behavior and use special purpose stabilization tricks to avoid the computation ending in a steaming heap of `nan`s. 

STensor takes care of these tricks in the background, so you don't have to. The goal is to make computations like the above _just work_, freeing the implementer to focus on more interesting problems. For the same example given above, we make a tiny change and see what happens:

```python
import torch
from stensor import stensor # STensor wrapper function

matrix = torch.eye(5) * 10
vector = torch.ones(5)
vector = stensor(vector)    # Conversion to STensor

for i in range(100):
    vector = matrix @ vector

output = torch.log10(vector)
print(output)
# >>> stensor([100., 100., 100., 100., 100.])
```

And _voilà_, we have the answer we wanted! We see that the output is no longer a Pytorch `Tensor`, but rather a `STensor`. The latter can be used exactly as the former, and conversion between the two occurs with the `stensor` function (`Tensor -> STensor`) and the `.torch()` method (`STensor -> Tensor`).

```python
x = stensor(torch.arange(5))
print(x)
# >>> stensor([0., 1., 2., 3., 4.])
print(x.torch())
# >>> tensor([0., 1., 2., 3., 4.])
```

## How STensor works

While the above example might look like magic, the underlying mechanism behind STensor is quite simple. Let's look at the massive vector produced by the iterated matrix-vector multiplication above

```python
print(vector)
# >>> stensor([inf, inf, inf, inf, inf])
# inf and/or zero entries may be artifact of conversion
# to Tensor, use repr to view underlying data

print(repr(vector))
# >>> STensor(data=
# tensor([0.5715, 0.5715, 0.5715, 0.5715, 0.5715]), scale=
# tensor([333.]))
```

Although the literal entries of the vector are too large to represent directly, the stensor itself consists of two tensors of reasonable size. The first "data" tensor gives a rescaled version of the stensor, and the second "scale" tensor gives a (base 2) logarithm-scale correction to the overall magnitude of the data. For any stensor `stens`, the tensor it encodes is given by:

```python
stens.torch() == stens.data * 2**stens.scale
```

While our earlier example showed `vector.scale` as a simple scalar, more sophisticated behavior is possible by feeding in a list of "stable" dimensions during the conversion to STensor:

```python
y = torch.randn(2, 3, 4)
y = stensor(y, stable_dims=[0, 1])
print(y.shape)
# >>> torch.Size([2, 3, 4])

print(y.scale.shape)
# >>> torch.Size([2, 3, 1])
```

In this case, each dimension-4 "fiber" of the data tensor (i.e. `y[i, j, :]`) will be associated with a separate scale factor, allowing for greater granularity when working with batched data that contains substantial variation in magnitude between different elements in the batch.

## What's the catch?

Nothing comes for free, and STensor is no exception. At present, the biggest downside of STensor is that it is still in an early stage of development. Many of the core Pytorch functions still have yet to be adapted for STensors, which means that feeding STensors into complex user-written functions will frequently lead to errors. While we eventually aim to make STensor a drop-in replacement for the Pytorch Tensor class, the massive size of the latter means this will take time.

More fundamentally, the use of extra scale information and the necessity of constantly rescaling imposes some extra computational overhead. While we aim to make this as small as possible, we nonetheless encourage users to do their own benchmarking, and let us know if they find any unexpected slowdowns while using STensor.

In the event that you encounter an issue with the library, please let us know on the [issues page](https://github.com/jemisjoky/STensor/issues) so we can prioritize it. Your feedback is vital for making this project grow!
