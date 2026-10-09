# sfGEMM

Fresh MIT CUDA C++17 implementation of row-major FP32 matrix multiplication:
`C = alpha * A * B + beta * C`. Include `src/kernel.cuh` and call
`sfGEMM::gemm(a, b, c, m, n, k, alpha, beta, stream)` with device pointers.
The call enqueues on the supplied CUDA stream and returns argument or launch
status. Completion errors belong to the caller's stream synchronization.

A is `[m,k]`, B is `[k,n]`, C is `[m,n]`, all contiguous FP32. A 16 by 16
shared-memory tile handles small products. When m and n are at least 128 and
k is at least 32, a 64 by 64 tile accumulates 16 outputs per thread in
registers and shares each input across more outputs. For aligned dimensions
(m divisible by 32, n by 64, k by 16) whose matrix element offsets fit in
32 bits and whose tile grid fits CUDA limits, a 32 by 64 output tile uses
eight register accumulators per thread. Its transposed, padded shared A
tile permits `float2` loads and shared B permits `float4` loads, reducing
shared load instructions. These vector accesses are internal shared-memory
accesses; external A/B/C need only ordinary float alignment. All paths use
FP32 `fmaf` accumulation without TF32 or reduced-precision inputs. Rectangular dimensions and
partial tiles are supported. Negative dimensions, missing required pointers,
extents beyond addressable storage, and exact A/C or B/C aliases are rejected.
Partially overlapping C/input ranges are unsupported. Zero m or n is a no-op;
zero k or alpha scales C without reading A/B. Beta zero never reads old C,
including when old C contains NaNs. Finite alpha/beta and finite A/B values
are the normal numerical contract; FP32 overflow is possible.

Portable source targets Hopper and Blackwell (`sm_90`, `sm_100`, `sm_103`,
`sm_120`, `sm_121`, including GB10). It uses CUDA cores, not an
architecture-specific tensor-core path. Compilation compatibility is separate
from hardware correctness and performance qualification. During matched
GB10 tuning with CUDA 13.0, the retained vector-load path at 512 x 512 x 512
took 49.236 us versus pedantic FP32 cuBLAS 28.888 us. This is 2.37x faster
than the initial implementation's 116.818 us on the same device, but still
1.70x slower than cuBLAS. The 64 x 64 x 64 path is unchanged. These
seven-sample CUDA-event medians exclude setup, allocations and transfers;
they do not qualify other sizes or devices. The suite records source-bound
qualification and benchmark evidence separately.

`tests/check.cu` compares GPU results to independent scalar double precision
matrix products on a nondefault stream. Cases cover scalar and rectangular
shapes, full FP32 input precision, dispatch boundaries, multiple register tiles, tails on every dimension,
alpha/beta behavior, NaN-filled prior C,
empty dimensions, invalid arguments, alias rejection, and output sentinels.
The executable prints one JSON verdict with per-case results and fails if any
case fails. A CUDA toolkit and GPU are required to run it.

Prior-source inventory: solidSF has generic tiled GEMM, mixed-format GEMV,
CUTLASS Hopper GEMM, and Blackwell block-scaled FP8 GEMM. Their precise paths
remain in the private suite inventory. They informed the capability map only;
no private source or third-party implementation is copied here. This package
does not provide FP16/BF16, FP8, strided/batched GEMM, autotuning, or CUTLASS.

## Standalone build

CUDA 13.0 or newer is required for the full target matrix.

```sh
make -j1
make test
# A single device family can be selected explicitly:
make ARCHS=90 BUILD_DIR=build-hopper
```

Default targets: H100/H200 (`sm_90`), B200 (`sm_100`), B300/GB300 (`sm_103`),
GB10 (`sm_121`), plus portable Hopper PTX. Do not add `--use_fast_math`;
that changes the numerical contract. Compile success and numerical
qualification are reported separately from performance benchmarking.

## Published source and GB10 measurements

The standalone source repository is [solidsf-inc/sfGEMM](https://github.com/solidsf-inc/sfGEMM). The [solidSF CUDA kernel catalog](https://solidsf.com/kernels) contains the formal report, all measured workloads and raw timing samples.

| Workload | SF median us | Reference median us | Reference / SF |
| --- | ---: | ---: | ---: |
| gemm fp32; 64 x 64 x 64 | 4.118 | 6.500 | 1.578x |
| gemm fp32; 512 x 512 x 512 | 49.236 | 28.888 | 0.587x |

References: cuBLAS cublasSgemm pedantic FP32 no TF32.

These are seven-sample medians of alternating CUDA-event measurements on one GB10 at stock clocks. Setup, allocations, transfers, plan creation and destructive-input resets are excluded; stream timing can include API enqueue gaps. Ratios above one mean SF is faster; near-unity medians do not establish a meaningful advantage. No aggregate or universal speedup is claimed.

All 16 package numerical/API cases passed on GB10. All four cubin targets and compute_90 PTX were verified. H100, H200, B200, B300 and GB300 hardware runtime and performance remain unmeasured.
