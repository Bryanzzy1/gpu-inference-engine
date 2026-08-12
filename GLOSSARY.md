
## Build / toolchain (build.md)
- **nvcc:** NVIDIA's CUDA compiler. On Windows it drives the MSVC host compiler (cl.exe),
  not MinGW g++, so `vcvars64.bat` must be loaded in the same shell first.
- **vcvars64.bat:** the VS Build Tools script that puts cl.exe + the MSVC env on PATH.
- **-arch=sm_89:** target the local Ada GPU (compute capability 8.9) natively, so the
  driver loads SASS directly instead of JIT-compiling PTX.
- **SASS / PTX:** SASS = the GPU's native machine code; PTX = the portable intermediate
  the driver JITs to SASS if no matching arch is present.
- **SM (streaming multiprocessor):** one physical core cluster; holds registers + shared
  memory and runs blocks. Compute capability (e.g. 8.9) describes an SM generation.
