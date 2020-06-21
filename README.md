
# Overview
This is an example of Multi-Layer Perceprton realisation on C++.

## Related libraries:
- Eigen - cpp library for vectors and matrix
- Qt - frontend cpp framework
- MNIST dataset

### Eigen
The library for working with matrices and vectors has chosen the Eigen package, the capabilities of which simultaneously satisfy the convenience and speed of development. The library supports all sizes of matrices, from small matrices of the fixed size to rather big and sparse matrices. Eigen supports all standard numeric types, including complex, integers, and easily extends to special numeric types. It supports various matrix decompositions and geometric features. The library modules provide many specialized functions, such as nonlinear optimization, matrix functions, polynomial calculator, FFT and much more. Explicit vectorization is performed for SSE 2/3/4, AVX, AVX2, FMA, AVX512, ARM NEON (32-bit and 64-bit), PowerPC AltiVec / VSX (32-bit and 64-bit), ZVector (s390x / zEC13 ) Fixed size matrices are fully optimized: dynamic memory allocation is avoided. For large matrices there is caching.

## Usage

Workspace - a grid of 28 by 28 cells, where each cell corresponds to a transparency from 0 to 1. When you click on any cell, it becomes black and transparency 1. When you click on the "Clear" button, the work area is cleared. To recognize an arbitrarily drawn number, follow these steps:
1. If necessary - clean the grid.
2. Draw a number.
3. Click the "Test against custom input" button.
4. Wait until the white cells turn gray.
To check the operation of the neural network on a random example from the standard set of handwritten numbers MNIST, you must click "Random Number" and wait for the result.

# Example

![Example-1](https://i.postimg.cc/sXrRmrtk/2.png)
![Example-2](https://i.postimg.cc/wvVnks5N/1.png)
