// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <climits>

namespace sfGEMM {
namespace detail {
inline bool extent_ok(int rows, int columns) {
    return static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns)
        <= static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(float);
}

static __global__ void scale_kernel(float* c, uint64_t count, float beta) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        c[i] = beta == 0.0f ? 0.0f : beta * c[i];
    }
}

static __global__ void gemm_kernel(const float* a, const float* b, float* c,
                                    int m, int n, int k, float alpha, float beta,
                                    uint64_t tile_columns, uint64_t tile_count) {
    __shared__ float as[16][16];
    __shared__ float bs[16][16];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    for (uint64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const uint64_t row = (tile / tile_columns) * 16 + ty;
        const uint64_t column = (tile % tile_columns) * 16 + tx;
        float sum = 0.0f;
        for (uint64_t base = 0; base < static_cast<uint64_t>(k); base += 16) {
            const uint64_t ak = base + tx;
            const uint64_t bk = base + ty;
            as[ty][tx] = row < static_cast<uint64_t>(m) && ak < static_cast<uint64_t>(k)
                ? a[row * k + ak] : 0.0f;
            bs[ty][tx] = bk < static_cast<uint64_t>(k) && column < static_cast<uint64_t>(n)
                ? b[bk * n + column] : 0.0f;
            __syncthreads();
#pragma unroll
            for (int inner = 0; inner < 16; ++inner)
                sum = fmaf(as[ty][inner], bs[inner][tx], sum);
            __syncthreads();
        }
        if (row < static_cast<uint64_t>(m) && column < static_cast<uint64_t>(n)) {
            const uint64_t index = row * n + column;
            const float prior = beta == 0.0f ? 0.0f : beta * c[index];
            c[index] = fmaf(alpha, sum, prior);
        }
        __syncthreads();
    }
}

static __global__ void register_gemm_kernel(const float* a, const float* b, float* c,
                                            int m, int n, int k, float alpha, float beta,
                                            uint64_t tile_columns, uint64_t tile_count) {
    __shared__ float as[64][16];
    __shared__ float bs[16][64];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int lane = ty * 16 + tx;
    for (uint64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const uint64_t row_base = (tile / tile_columns) * 64;
        const uint64_t column_base = (tile % tile_columns) * 64;
        float sums[4][4] = {};
        for (uint64_t base = 0; base < static_cast<uint64_t>(k); base += 16) {
#pragma unroll
            for (int load = 0; load < 4; ++load) {
                const int offset = lane + load * 256;
                const uint64_t row = row_base + offset / 16;
                const uint64_t ak = base + offset % 16;
                const uint64_t bk = base + offset / 64;
                const uint64_t column = column_base + offset % 64;
                as[offset / 16][offset % 16] = row < static_cast<uint64_t>(m)
                    && ak < static_cast<uint64_t>(k) ? a[row * k + ak] : 0.0f;
                bs[offset / 64][offset % 64] = bk < static_cast<uint64_t>(k)
                    && column < static_cast<uint64_t>(n) ? b[bk * n + column] : 0.0f;
            }
            __syncthreads();
#pragma unroll
            for (int inner = 0; inner < 16; ++inner) {
                float av[4], bv[4];
#pragma unroll
                for (int index = 0; index < 4; ++index) {
                    av[index] = as[ty + index * 16][inner];
                    bv[index] = bs[inner][tx + index * 16];
                }
#pragma unroll
                for (int row = 0; row < 4; ++row)
#pragma unroll
                    for (int column = 0; column < 4; ++column)
                        sums[row][column] = fmaf(av[row], bv[column], sums[row][column]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const uint64_t output_row = row_base + ty + row * 16;
#pragma unroll
            for (int column = 0; column < 4; ++column) {
                const uint64_t output_column = column_base + tx + column * 16;
                if (output_row < static_cast<uint64_t>(m)
                    && output_column < static_cast<uint64_t>(n)) {
                    const uint64_t index = output_row * n + output_column;
                    const float prior = beta == 0.0f ? 0.0f : beta * c[index];
                    c[index] = fmaf(alpha, sums[row][column], prior);
                }
            }
        }
        __syncthreads();
    }
}

static __global__ void vector_gemm_kernel(const float* a, const float* b, float* c,
                                          uint32_t n, uint32_t k, float alpha, float beta) {
    __shared__ __align__(16) float as[16][36];
    __shared__ __align__(16) float bs[16][64];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int lane = ty * 16 + tx;
    const uint32_t row_base = blockIdx.y * 32;
    const uint32_t column_base = blockIdx.x * 64;
    float sums[2][4] = {};
    for (uint32_t base = 0; base < k; base += 16) {
#pragma unroll
        for (int offset = lane; offset < 32 * 16; offset += 256)
            as[offset % 16][offset / 16]
                = a[(row_base + offset / 16) * k + base + offset % 16];
#pragma unroll
        for (int offset = lane; offset < 16 * 64; offset += 256)
            bs[offset / 64][offset % 64]
                = b[(base + offset / 64) * n + column_base + offset % 64];
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < 16; ++inner) {
            const float2 av = *reinterpret_cast<const float2*>(&as[inner][ty * 2]);
            const float4 bv = *reinterpret_cast<const float4*>(&bs[inner][tx * 4]);
            sums[0][0] = fmaf(av.x, bv.x, sums[0][0]);
            sums[0][1] = fmaf(av.x, bv.y, sums[0][1]);
            sums[0][2] = fmaf(av.x, bv.z, sums[0][2]);
            sums[0][3] = fmaf(av.x, bv.w, sums[0][3]);
            sums[1][0] = fmaf(av.y, bv.x, sums[1][0]);
            sums[1][1] = fmaf(av.y, bv.y, sums[1][1]);
            sums[1][2] = fmaf(av.y, bv.z, sums[1][2]);
            sums[1][3] = fmaf(av.y, bv.w, sums[1][3]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int row = 0; row < 2; ++row)
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const uint32_t index = (row_base + ty * 2 + row) * n
                + column_base + tx * 4 + column;
            const float prior = beta == 0.0f ? 0.0f : beta * c[index];
            c[index] = fmaf(alpha, sums[row][column], prior);
        }
}
} // namespace detail

// Row-major C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n].
inline cudaError_t gemm(const float* a, const float* b, float* c,
                        int m, int n, int k, float alpha = 1.0f,
                        float beta = 0.0f, cudaStream_t stream = nullptr) {
    if (m < 0 || n < 0 || k < 0) return cudaErrorInvalidValue;
    if (m == 0 || n == 0) return cudaSuccess;
    if (!c || !detail::extent_ok(m, n)) return cudaErrorInvalidValue;
    if (k == 0 || alpha == 0.0f) {
        const uint64_t count = static_cast<uint64_t>(m) * n;
        const uint64_t wanted = (count + 255) / 256;
        const unsigned blocks = static_cast<unsigned>(wanted < 65535 ? wanted : 65535);
        detail::scale_kernel<<<blocks, 256, 0, stream>>>(c, count, beta);
        return cudaGetLastError();
    }
    if (!a || !b || a == c || b == c || !detail::extent_ok(m, k)
        || !detail::extent_ok(k, n)) return cudaErrorInvalidValue;
    const bool register_blocked = m >= 128 && n >= 128 && k >= 32;
    if (register_blocked && m % 32 == 0 && n % 64 == 0 && k % 16 == 0
        && m / 32 <= 65535 && n / 64 <= 65535
        && static_cast<uint64_t>(m) * n <= UINT32_MAX
        && static_cast<uint64_t>(m) * k <= UINT32_MAX
        && static_cast<uint64_t>(k) * n <= UINT32_MAX) {
        detail::vector_gemm_kernel<<<dim3(n / 64, m / 32), dim3(16, 16), 0, stream>>>(
            a, b, c, n, k, alpha, beta);
        return cudaGetLastError();
    }
    const uint64_t tile_size = register_blocked ? 64 : 16;
    const uint64_t columns = (static_cast<uint64_t>(n) + tile_size - 1) / tile_size;
    const uint64_t tiles = ((static_cast<uint64_t>(m) + tile_size - 1) / tile_size) * columns;
    const unsigned blocks = static_cast<unsigned>(tiles < 65535 ? tiles : 65535);
    if (register_blocked)
        detail::register_gemm_kernel<<<blocks, dim3(16, 16), 0, stream>>>(
            a, b, c, m, n, k, alpha, beta, columns, tiles);
    else
        detail::gemm_kernel<<<blocks, dim3(16, 16), 0, stream>>>(
            a, b, c, m, n, k, alpha, beta, columns, tiles);
    return cudaGetLastError();
}
} // namespace sfGEMM
