# AI Coding Agent Instructions for MLX

These notes make AI agents immediately productive in this repo by codifying MLX’s architecture, workflows, and conventions. Keep changes minimal, aligned with existing patterns, and validated via tests.

## Big Picture
- **Core library (`mlx/`)**: Modern C++17 array framework with lazy execution, dynamic graphs, unified memory, and multi-device backends. Key areas:
  - **Arrays & dtypes**: `array.{h,cpp}`, `dtype.{h,cpp}`, `dtype_utils.{h,cpp}`
  - **Ops & transforms**: `ops.{h,cpp}`, `transforms.{h,cpp}`, `autograd` lives in transforms
  - **Linalg / FFT / Random**: `linalg.{h,cpp}`, `fft.{h,cpp}`, `random.{h,cpp}`
  - **Compilation & export**: `compile.{h,cpp}`, `export.{h,cpp}`
  - **Device & scheduler**: `device.{h,cpp}`, `scheduler.{h,cpp}` – execution and streams
  - **Utils & memory**: `allocator.{h,cpp}`, `utils.{h,cpp}`, `memory.h`
- **Backends**: CPU always available; Metal (macOS) and CUDA (Linux) are optional. See `mlx/backend/` and CMake options.
- **Python bindings**: Optional native bindings under `python/src` exposed via `pip install mlx[...]` on users’ systems; MLX Python mirrors NumPy/PyTorch ergonomics.
- **Testing**: C++ tests via `doctest` in `tests/`; Python tests in `python/tests/`. Benchmarks and examples live under `benchmarks/` and `examples/`.

## Build & Test Workflows
- **CMake build (C++ library + tests)**:
  - Configure options (default: CPU and Metal on macOS; CUDA off):
    - `MLX_BUILD_TESTS` (ON), `MLX_BUILD_EXAMPLES` (ON), `MLX_BUILD_BENCHMARKS` (OFF)
    - Backends: `MLX_BUILD_CPU` (ON), `MLX_BUILD_METAL` (macOS only), `MLX_BUILD_CUDA` (Linux optional)
    - Python bindings: `MLX_BUILD_PYTHON_BINDINGS` (OFF by default)
  - Typical Linux CPU-only + tests:
    ```bash
    cmake -S . -B build -DMLX_BUILD_TESTS=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_METAL=OFF
    cmake --build build -j
    ctest --test-dir build --output-on-failure
    ```
- **Enable CUDA backend (Linux)**:
  ```bash
  cmake -S . -B build -DMLX_BUILD_CUDA=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```
  The C++ tests add GPU test sources when CUDA/Metal is enabled. CCCL headers are wired for JIT in `tests/CMakeLists.txt`.
- **Python tests**:
  ```bash
  pip install -e .[cpu]  # or .[cuda]
  pytest -q python/tests
  ```
  Python tests cover arrays, ops, autograd, linalg, fft, nn, optimizers, etc.

- **Benchmarks**:
  - C++ benchmarks live in `benchmarks/cpp/` (e.g., `single_ops.cpp`, `autograd.cpp`). Enable via CMake:
    ```bash
    cmake -S . -B build -DMLX_BUILD_BENCHMARKS=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF
    cmake --build build -j
    # Binaries are placed under build/benchmarks/cpp
    ./build/benchmarks/cpp/single_ops
    ```
  - Python benchmarks live under `benchmarks/python/` and `benchmarks/numpy/`. Run directly:
    ```bash
    # MLX Python installed (cpu or cuda extra)
    pip install -e .[cpu]
    python benchmarks/python/conv_bench.py
    python benchmarks/numpy/single_ops.py
    ```
  - Some scripts accept flags for devices and sizes; check each file’s `if __name__ == '__main__'` for arguments.

## Microbenchmarks: MLX vs PyTorch
Microbenchmarks for MLX and PyTorch are in `benchmarks/python/comparative/`.

- **Run a single MLX or PyTorch microbenchmark:**
  ```bash
  # MLX example: sum across axis 2 on CPU
  python benchmarks/python/comparative/bench_mlx.py sum_axis --size 8x1024x128 --axis 2 --cpu

  # PyTorch example: same operation
  python benchmarks/python/comparative/bench_torch.py sum_axis --size 8x1024x128 --axis 2 --cpu
  ```
  - The `benchmark` argument selects the operation (see script for options).
  - Use `--size` to set tensor shape (e.g., `8x1024x128`), `--axis` for reduction axis, `--cpu` to force CPU, and `--fused` for fused ops where available.
  - `--print-pid` prints the process ID and waits for a keypress (useful for attaching a debugger).
  - Other options: `--dtype`, `--transpose`, see script help (`-h`).

- **Compare MLX and PyTorch automatically:**
  ```bash
  python benchmarks/python/comparative/compare.py
  ```
  This script runs a suite of benchmarks and reports speedup/regression.

- **Dev tips:**
  - Both `bench_mlx.py` and `bench_torch.py` are structured for easy extension: add new ops as functions and register them in the main block.
  - Each script runs the selected op 100 times (10 warmup) and reports elapsed time.
  - For debugging, use `--print-pid` and attach a debugger before pressing enter.
  - See `benchmarks/python/comparative/README.md` for more usage notes.


## Build From Source (doc-backed)
These steps are distilled from the project docs: https://ml-explore.github.io/mlx/build/html/install.html

### Build requirements
- C++17-capable compiler (Clang/GCC), `cmake` >= 3.25, `make`.
- On macOS: Xcode >= 15.0 and macOS SDK >= 14.0; ensure your shell runs natively on `arm` (not Rosetta).

### Linux prerequisites (CPU or CUDA)
Install BLAS/LAPACK headers (Ubuntu example):
```bash
apt-get update -y
apt-get install libblas-dev liblapack-dev liblapacke-dev -y
```

### CUDA prerequisites (Linux)
Install CUDA toolkit and BLAS/LAPACK headers (Ubuntu example):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -y
apt-get -y install cuda-toolkit-12-9
apt-get install libblas-dev liblapack-dev liblapacke-dev libcudnn9-dev-cuda-12 -y
```

---

### Python API: build and install from source
Clone and install:
```bash
git clone git@github.com:ml-explore/mlx.git mlx && cd mlx
pip install .
```
For development (editable install, dev dependencies, faster local builds, tests, and IDE stubs):
```bash
pip install -e ".[dev]"
python setup.py build_ext --inplace
python -m unittest discover python/tests
python setup.py generate_stubs  # optional
```

---

### C++ API: build and install from source
Clone, build, test, and install:
```bash
git clone git@github.com:ml-explore/mlx.git mlx && cd mlx
mkdir -p build && cd build
cmake .. && make -j
make test
sudo make install
```
To minimize binary size, add flags:
```bash
cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel -DBUILD_SHARED_LIBS=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_SAFETENSORS=OFF -DMLX_BUILD_GGUF=OFF -DMLX_METAL_JIT=ON
```

---

### CUDA build (Linux)
When building either the Python or C++ APIs, pass the CMake flag `-DMLX_BUILD_CUDA=ON` (or set `CMAKE_ARGS="-DMLX_BUILD_CUDA=ON"` for `pip install`).
Example for Python:
```bash
CMAKE_ARGS="-DMLX_BUILD_CUDA=ON" pip install -e ".[dev]"
```
Example for C++:
```bash
mkdir -p build && cd build
cmake .. -DMLX_BUILD_CUDA=ON && make -j
```

---

### Troubleshooting highlights
- If Metal tools are missing on macOS: `xcode-select --install` and `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`.
- If your shell reports `uname -p` as `x86` on Apple silicon, disable Rosetta for your terminal app and restart; then wipe `build/` and rebuild.


## Conventions & Patterns
- **Lazy + dynamic graphs**: Ops compose into graphs; materialization occurs at use. Avoid forcing eager sync unless required by API contract.
- **Unified memory model**: Arrays can be used across devices without explicit transfers; device selection is via `device.{h,cpp}`. Respect this in new ops.
- **Shape-agnostic compute**: Changing shapes should not trigger recompilation; prefer kernels that handle variable shapes and strides.
- **Header-first APIs**: Public C++ APIs live in headers (`*.h`) paired with minimal, focused implementations in `*.cpp`. Keep symbols stable and documented.
- **Error handling**: Use existing utilities in `utils.{h,cpp}`; follow existing return/throw patterns from similar functions rather than introducing new ones.
- **Testing style**:
  - C++: `doctest`-based unit tests in `tests/*.cpp`, aggregated via `tests/tests.cpp` and `doctest_discover_tests`.
  - Python: `pytest` tests in `python/tests`, mirroring NumPy/PyTorch semantics and MLX specifics.
- **Performance and BLAS/LAPACK**: CPU backend links LAPACK/BLAS or OpenBLAS (optional source build). See `CMakeLists.txt` for vendor detection and include dirs.

## Adding/Modifying Ops (example-driven)
- When implementing a new op (e.g., `searchsorted`):
  - Add public declaration in `ops.h` and implementation in `ops.cpp` following existing function signatures and dtype dispatch in `dtype_utils.*`.
  - If vectorized or autograd-aware, extend `transforms.{h,cpp}` and `autograd` patterns used by `vmap`, `grad`, and `custom_vjp_tests.cpp`.
  - Integrate device-specific paths via existing fast primitives or backend hooks in `backend/` as done for `linalg`, `fft`, or `random`.
  - Add C++ unit tests under `tests/ops_tests.cpp` and Python tests under `python/tests/test_ops.py` with concrete shape/dtype/device cases.

## External Integrations
- **FetchContent**: Third-party deps managed via CMake `FetchContent` (e.g., `fmt`, `nlohmann/json`, `doctest`). Prefer this mechanism for new C++ deps.
- **Metal (macOS)**: Requires macOS SDK ≥ 14.0 and `metal-cpp`. Auto-fetched and installed; guarded by `MLX_BUILD_METAL` and `MLX_METAL_DEBUG`.
- **CUDA (Linux)**: Optional; enables CUDA language and GPU test sources. CCCL path configured in tests. Keep CUDA-specific code isolated under backend.
- **Python bindings**: Built via `nanobind` when `MLX_BUILD_PYTHON_BINDINGS=ON` and `python/src` is included by CMake.

## Where to Look First
- C++ API surface: `mlx/ops.h`, `mlx/array.h`, `mlx/device.h`, `mlx/transforms.h`
- Reference implementations: `mlx/ops.cpp`, `mlx/linalg.cpp`, `mlx/random.cpp`
- Build config and options: `CMakeLists.txt`, `cmake/*.cmake`
- Tests as usage guides: `tests/*.cpp`, `python/tests/*.py`
- Python packaging: `setup.py`, `pyproject.toml`, `python/mlx/`

## PR Hygiene for Agents
- Keep changes focused; mirror surrounding style and naming.
- Update or add tests near modified components; run C++ and Python tests.
- Avoid introducing new global patterns; re-use existing utility layers.
- Document any new public API in headers and ensure compilation across configured backends.

If any section is unclear or missing details (e.g., a specific backend path or test expectation), tell us what you need and we’ll iterate.