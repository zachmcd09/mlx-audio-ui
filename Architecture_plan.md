# MLX-Audio Cross-Platform Optimization Architecture Plan

## 1. Overview

This document outlines the architectural plan to optimize the `mlx-audio` library for high performance on Apple Silicon devices (leveraging Metal, MPS, and ANE) while establishing a foundation for future cross-platform compatibility with NVIDIA GPUs (CUDA) and Windows ARM devices (DirectML/Vulkan).

The core strategy involves:
1.  **Abstraction:** Defining a `ComputeBackend` interface to decouple core audio algorithms from hardware-specific implementations.
2.  **Prioritized Optimization:** Focusing initially on maximizing performance on Apple Silicon using native frameworks.
3.  **Modular Backends:** Implementing separate backends for different hardware targets (Metal, CUDA, DirectML/Vulkan).
4.  **Comprehensive Testing:** Ensuring numerical correctness and performance across all supported platforms.

## 2. High-Level Plan (Phased Approach)

```mermaid
graph TD
    A[Start: Current MLX-Audio Repo] --> B(Phase 1: Foundation & Abstraction);
    B --> C(Phase 2: Apple Silicon Optimization - Metal/MPS/ANE);
    C --> D(Phase 3: Cross-Platform Backend - CUDA);
    D --> E(Phase 4: Cross-Platform Backend - Windows ARM/DirectML);
    E --> F(Phase 5: Testing, Benchmarking & Refinement);
    F --> G(Goal: Optimized & Cross-Platform Library);

    subgraph Phase 1: Foundation & Abstraction
        B1[Define ComputeBackend Interface];
        B2[Refactor Core Algorithms (LSTM, Conv) using Interface];
        B3[Implement Default MLX Backend];
    end

    subgraph Phase 2: Apple Silicon Optimization
        C1[Implement MetalComputeBackend];
        C2[Optimize LSTM: MPS Matrix Accel (Mixed Precision)];
        C3[Optimize Convolutions: MPSCNN Kernels];
        C4[Memory Management: MTLHeap Recycling];
        C5[Explore ANE Offload via MPSGraph];
        C6[Custom Metal Kernels for Non-MPS Ops (Conflict Avoidance)];
    end

    subgraph Phase 3: Cross-Platform Backend - CUDA
        D1[Implement CUDABackend];
        D2[CUDA LSTM Kernels (cuBLAS/cuDNN, Mixed Precision)];
        D3[CUDA Conv Kernels (cuDNN)];
        D4[Leverage Unified Memory];
    end

    subgraph Phase 4: Cross-Platform Backend - Windows ARM/DirectML
        E1[Implement DirectMLBackend (or Vulkan)];
        E2[Map Operations to DirectML/Vulkan Equivalents];
        E3[Focus on Compatibility First];
    end

    subgraph Phase 5: Testing, Benchmarking & Refinement
        F1[Cross-Platform Unit & Integration Tests];
        F2[Performance Benchmarking Suite (Apple Silicon, NVIDIA, Win ARM)];
        F3[Numerical Precision Validation];
        F4[Iterative Refinement based on Results];
    end

```

## 3. Detailed Phase Breakdown

### Phase 1: Foundation & Abstraction (Cross-Platform)

*   **Goal:** Decouple algorithms from hardware, establish a common interface.
*   **Actions:**
    *   **Define `ComputeBackend` Protocol:**
        *   Location: `mlx_audio/backend/base.py` (new file/directory).
        *   Specify abstract methods for essential operations:
            *   `matrix_multiply(a, b, transpose_a=False, transpose_b=False)`
            *   `convolution(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`
            *   `lstm_cell(input, hidden_state, cell_state, weights_ih, weights_hh, bias_ih=None, bias_hh=None)`
            *   Activation functions (`relu`, `sigmoid`, `tanh`, etc.)
            *   Pooling operations (`max_pool`, `avg_pool`)
            *   Normalization layers (`layer_norm`, `batch_norm`)
            *   Tensor manipulation (`reshape`, `transpose`, `concat`, `pad`)
        *   Use Python's `typing.Protocol` or `abc.ABC`.
    *   **Refactor Core Algorithms:**
        *   Identify performance-critical modules (e.g., `encodec.py`, `llama.py`, `bark.py`, `sesame/model.py`).
        *   Modify these modules to accept a `ComputeBackend` instance during initialization or method calls.
        *   Replace direct MLX calls with calls to the backend interface methods (e.g., `backend.matrix_multiply(...)` instead of `mx.matmul(...)`).
    *   **Implement Default MLX Backend:**
        *   Location: `mlx_audio/backend/mlx_backend.py`.
        *   Create a concrete class `MLXBackend` implementing `ComputeBackend`.
        *   Implement interface methods using standard `mlx.core` and `mlx.nn` functions.
        *   This ensures the library remains functional during the transition.

### Phase 2: Apple Silicon Optimization (Metal/MPS/ANE)

*   **Goal:** Achieve near-native performance on Apple Silicon.
*   **Actions:**
    *   **Implement `MetalComputeBackend`:**
        *   Location: `mlx_audio/backend/metal_backend.py`.
        *   Requires bridging Python to Metal (e.g., using `ctypes`, `Cython`, or potentially a dedicated Swift/Objective-C++ module compiled via `setup.py`).
        *   Initialize `MTLDevice`, `MTLCommandQueue`.
    *   **Optimize LSTM with MPS:**
        *   Implement `lstm_cell` using `MPSRNNMatrixTrainingLayer` or equivalent MPS primitives.
        *   Prioritize **Mixed Precision:** Use `dataType: .float16` for inputs/weights and `.float32` for accumulation within MPS kernels where possible to leverage tensor cores.
    *   **Optimize Convolutions with MPS:**
        *   Implement `convolution` using `MPSCNNConvolution`. Configure data types for mixed precision if beneficial.
    *   **Optimize Matrix Multiplication with MPS:**
        *   Implement `matrix_multiply` using `MPSMatrixMultiplication`, again favoring mixed precision.
    *   **Memory Management:**
        *   Utilize `MTLHeap` for allocating Metal buffers (`MTLBuffer`) to reduce allocation overhead and enable efficient reuse, especially for intermediate tensors within complex operations.
        *   Manage buffer lifetimes carefully.
    *   **Custom Metal Kernels (Selective):**
        *   For operations not well-covered by MPS or requiring specific optimizations (e.g., unique activation functions, complex custom layers).
        *   Apply **Bank Conflict Avoidance** techniques (prime strides, padding, warp-centric addressing) discussed previously.
        *   Write kernels in Metal Shading Language (.metal files) and compile them at runtime or build time.
    *   **Explore ANE Offload (Exploratory):**
        *   Use `MPSGraph` to define parts of the computation graph (especially recurrent layers like LSTM).
        *   Compile the graph targeting the ANE (`options.defaultDevice = device.mpsDevice`).
        *   Benchmark performance and power usage compared to GPU execution. This is most beneficial for sustained workloads where the ANE's efficiency shines.

### Phase 3: Cross-Platform Backend - CUDA

*   **Goal:** Provide high-performance execution on NVIDIA GPUs.
*   **Actions:**
    *   **Implement `CUDABackend`:**
        *   Location: `mlx_audio/backend/cuda_backend.py`.
        *   Requires bridging Python to CUDA (e.g., using `CuPy`, `PyCUDA`, or custom C++/CUDA extensions).
    *   **Leverage CUDA Libraries:**
        *   Implement `matrix_multiply` using `cuBLAS`.
        *   Implement `convolution` using `cuDNN`.
        *   Implement `lstm_cell` using `cuDNN`'s RNN functions or optimized custom kernels if necessary.
        *   Mirror the mixed-precision strategy (FP16 compute with FP32 accumulation) using Tensor Cores where available.
    *   **Memory Management:**
        *   Utilize **CUDA Unified Memory** (`cudaMallocManaged`) where appropriate to simplify data transfers between CPU and GPU, especially for smaller or less frequently accessed tensors. Manage explicit transfers (`cudaMemcpy`) for performance-critical paths.

### Phase 4: Cross-Platform Backend - Windows ARM/DirectML (Future Scope)

*   **Goal:** Ensure architectural readiness for Windows on ARM.
*   **Actions:**
    *   **Interface Design:** Ensure the `ComputeBackend` interface remains generic enough not to preclude DirectML or Vulkan Compute implementations. Avoid Metal/CUDA-specific data types or concepts in the interface definition.
    *   **Future Implementation:**
        *   Create `DirectMLBackend` (or `VulkanComputeBackend`).
        *   Requires bridging to the respective APIs (e.g., using C++/WinRT for DirectML, or Vulkan SDK bindings).
        *   Map interface operations to DirectML operators or Vulkan compute shaders.
        *   Initial focus should be on functional correctness over peak performance.

### Phase 5: Testing, Benchmarking & Refinement

*   **Goal:** Ensure correctness, performance, and robustness across all platforms.
*   **Actions:**
    *   **Backend Dispatch Logic:**
        *   Implement a factory function or context manager in `mlx_audio/backend/__init__.py` to detect available hardware and select the most appropriate backend (Metal > CUDA > MLX default). Allow manual override via environment variables or configuration.
    *   **Comprehensive Test Suite (`tests/`):**
        *   **Unit Tests:** Test individual backend operations against reference NumPy/MLX implementations for numerical accuracy (using `np.allclose` with appropriate tolerances for different precisions). Parameterize tests to run across all available backends.
        *   **Integration Tests:** Test full audio processing pipelines (e.g., encodec encode/decode) end-to-end on each backend.
        *   **Precision Tests:** Specifically validate mixed-precision results against FP32 references.
    *   **Performance Benchmarking Suite:**
        *   Create scripts (`benchmarks/`) to measure execution time for key operations and end-to-end pipelines on target hardware (M-series Mac, NVIDIA GPU, eventually Win ARM).
        *   Measure metrics like latency, throughput, and potentially memory bandwidth usage.
    *   **Iterative Refinement:** Use benchmarking results and profiling data (Xcode Instruments for Metal, Nsight for CUDA) to identify bottlenecks and iteratively refine backend implementations.

## 4. Future Considerations

*   **Quantization:** Explore INT8 quantization support within backends (MPS/CUDA/DirectML offer relevant tools).
*   **Sparse Operations:** Investigate support for sparse weights/activations if models benefit.
*   **Asynchronous Execution:** Refactor backend calls to be potentially asynchronous for better pipeline parallelism.
*   **Build System:** Update `setup.py` to handle conditional compilation of backend extensions.
