# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-mlx (v0.1.0)

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-mlx/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

[Zero-dependency](https://en.wikipedia.org/wiki/Dependency_hell) pure [Python](https://www.python.org/) implementation of the [Apple MLX](https://github.com/ml-explore/mlx) API surface. 

## Why does this project exist?

`zero-mlx` is a core frontend component of the larger **Abstract ML Machine Ecosystem**. The broader ecosystem is designed to solve the $N \times M$ translation problem in [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning). Instead of writing bespoke translators from every ML framework ([JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/), [Keras](https://keras.io/), [MLX](https://github.com/ml-explore/mlx)) to every target hardware backend ([WASM](https://webassembly.org/), [WebGPU](https://www.w3.org/TR/webgpu/), [TensorRT](https://developer.nvidia.com/tensorrt)), we trace $N$ frontends into a strictly defined [Intermediate Representation](https://en.wikipedia.org/wiki/Intermediate_representation) (IR, via `ml-switcheroo-ir`), which is then consumed by $M$ backends.

This achieves a complete [source-to-source](https://en.wikipedia.org/wiki/Source-to-source_compiler) and source-to-browser compilation pipeline utilizing **strictly zero external dependencies**, relying solely on the [Python Standard Library](https://docs.python.org/3/library/) and [`numpy`](https://numpy.org/) for eager evaluations.

Specifically, `zero-mlx`:
- **Mimics MLX Semantics:** It replicates the eager, [object-oriented](https://en.wikipedia.org/wiki/Object-oriented_programming), and stateful semantics of Apple's MLX framework.
- **State Functionalization:** Under the hood, it seamlessly integrates with the `ml-switcheroo-compiler` to dynamically lift mutable MLX-style states (like [neural network](https://en.wikipedia.org/wiki/Neural_network) parameters) into purely [functional](https://en.wikipedia.org/wiki/Functional_programming) graph inputs and outputs via an internal `lift_state` pass.
- **Cross-Platform Compatibility:** By rewriting MLX code in pure [Python](https://www.python.org/), it enables execution, tracing, and compilation of [MLX](https://github.com/ml-explore/mlx) models on non-Apple hardware (like browsers via [WASM](https://webassembly.org/)/[WebGPU](https://www.w3.org/TR/webgpu/), or native generic targets) without needing the core [C++](https://isocpp.org/)/[Metal](https://developer.apple.com/metal/) Apple-specific MLX binaries.
- **Zero-Dependency Tracing:** Uses ProxyTensors to overload Python math dunders, intercept eager operations, and record them to a TracerTape without needing heavy framework dependencies.

In short, `zero-mlx` lets you write and execute standard [MLX](https://github.com/ml-explore/mlx) models, while transparently bridging them to universal backends and environments through the Abstract ML compiler stack.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
