# zero-mlx (v0.1.0)

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-mlx/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

Zero-dependency pure Python implementation of the [Apple MLX](https://github.com/ml-explore/mlx) API surface. 

## Why does this project exist?

`zero-mlx` is a core frontend component of the larger **Abstract ML Machine Ecosystem**. The broader ecosystem is designed to solve the $N \times M$ translation problem in Machine Learning. Instead of writing bespoke translators from every ML framework (JAX, PyTorch, Keras, MLX) to every target hardware backend (WASM, WebGPU, TensorRT), we trace $N$ frontends into a strictly defined Intermediate Representation (IR, via `ml-switcheroo-ir`), which is then consumed by $M$ backends.

This achieves a complete source-to-source and source-to-browser compilation pipeline utilizing **strictly zero external dependencies**, relying solely on the Python Standard Library and `numpy` for eager evaluations.

Specifically, `zero-mlx`:
- **Mimics MLX Semantics:** It replicates the eager, object-oriented, and stateful semantics of Apple's MLX framework.
- **State Functionalization:** Under the hood, it seamlessly integrates with the `ml-switcheroo-compiler` to dynamically lift mutable MLX-style states (like neural network parameters) into purely functional graph inputs and outputs via an internal `lift_state` pass.
- **Cross-Platform Compatibility:** By rewriting MLX code in pure Python, it enables execution, tracing, and compilation of MLX models on non-Apple hardware (like browsers via WASM/WebGPU, or native generic targets) without needing the core C++/Metal Apple-specific MLX binaries.
- **Zero-Dependency Tracing:** Uses ProxyTensors to overload Python math dunders, intercept eager operations, and record them to a TracerTape without needing heavy framework dependencies.

In short, `zero-mlx` lets you write and execute standard MLX models, while transparently bridging them to universal backends and environments through the Abstract ML compiler stack.

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
