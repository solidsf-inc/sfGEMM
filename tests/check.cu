// SPDX-License-Identifier: MIT
#include "../src/kernel.cuh"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void require(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    float* data = nullptr;
    size_t count;
    explicit Buffer(size_t size) : count(size) { require(cudaMalloc(&data, count * sizeof(float))); }
    ~Buffer() { cudaFree(data); }
    void write(const std::vector<float>& values, cudaStream_t stream) {
        require(cudaMemcpyAsync(data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    std::vector<float> read(cudaStream_t stream) {
        std::vector<float> result(count);
        require(cudaMemcpyAsync(result.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        require(cudaStreamSynchronize(stream));
        return result;
    }
};

static bool product_case(int m, int n, int k, float alpha, float beta, cudaStream_t stream,
                         bool full_precision_inputs = false) {
    std::vector<float> a(m * k), b(k * n), c(m * n + 9, 771.0f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (static_cast<int>(i % 17) - 8)
        * (full_precision_inputs ? 0.0001220703125f : 0.125f)
        + (full_precision_inputs ? 1.0001220703125f : 0.0f);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (static_cast<int>(i % 13) - 6)
        * (full_precision_inputs ? 0.000244140625f : 0.25f)
        + (full_precision_inputs ? 1.000244140625f : 0.0f);
    for (int i = 0; i < m * n; ++i)
        c[i + 1] = beta == 0.0f ? std::numeric_limits<float>::quiet_NaN() : (i % 5 - 2) * 0.5f;
    Buffer da(a.size()), db(b.size()), dc(c.size());
    da.write(a, stream); db.write(b, stream); dc.write(c, stream);
    require(sfGEMM::gemm(da.data, db.data, dc.data + 1, m, n, k, alpha, beta, stream));
    const auto result = dc.read(stream);
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < n; ++column) {
            double sum = 0.0;
            for (int inner = 0; inner < k; ++inner)
                sum += static_cast<double>(a[row * k + inner]) * b[inner * n + column];
            const int index = 1 + row * n + column;
            const double expected = alpha * sum + (beta == 0.0f ? 0.0 : beta * c[index]);
            if (!std::isfinite(result[index])
                || std::abs(result[index] - expected) > 2e-5 * (1.0 + std::abs(expected)))
                return false;
        }
    }
    if (result[0] != 771.0f) return false;
    for (size_t i = 1 + m * n; i < result.size(); ++i) if (result[i] != 771.0f) return false;
    return true;
}

static bool scale_case(bool zero_k, float beta, cudaStream_t stream) {
    std::vector<float> original(35 + 9, 771.0f);
    for (size_t i = 0; i < 35; ++i) original[i + 1] = beta == 0.0f
        ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(i) - 9.0f;
    Buffer output(original.size()); output.write(original, stream);
    require(sfGEMM::gemm(nullptr, nullptr, output.data + 1, 5, 7, zero_k ? 0 : 19,
                         zero_k ? 1.0f : 0.0f, beta, stream));
    const auto result = output.read(stream);
    for (size_t i = 0; i < result.size(); ++i) {
        const float expected = i == 0 || i >= 36 ? 771.0f
            : beta == 0.0f ? 0.0f : beta * original[i];
        if (result[i] != expected) return false;
    }
    return true;
}

int main() {
    std::vector<std::pair<std::string, bool>> cases;
    cudaStream_t stream = nullptr;
    try {
        cases.emplace_back("invalid_dimensions", sfGEMM::gemm(nullptr, nullptr, nullptr, -1, 2, 3) == cudaErrorInvalidValue
            && sfGEMM::gemm(nullptr, nullptr, nullptr, 1, 2, 3) == cudaErrorInvalidValue);
        cases.emplace_back("empty_dimensions", sfGEMM::gemm(nullptr, nullptr, nullptr, 0, 7, 11) == cudaSuccess
            && sfGEMM::gemm(nullptr, nullptr, nullptr, 7, 0, 11) == cudaSuccess);
        require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cases.emplace_back("scalar_beta_zero_nan_prior", product_case(1, 1, 1, 1.0f, 0.0f, stream));
        cases.emplace_back("rectangular_three_tails", product_case(19, 35, 23, 0.75f, -0.5f, stream));
        cases.emplace_back("aligned_rows_k_tail", product_case(64, 17, 49, -1.25f, 0.0f, stream));
        cases.emplace_back("register_tile_beta_zero_nan_prior", product_case(128, 128, 32, 1.0f, 0.0f, stream));
        cases.emplace_back("register_full_fp32_input_precision", product_case(128, 128, 32, 1.0f, 0.0f, stream, true));
        cases.emplace_back("aligned_multiple_tiles_alpha_beta", product_case(192, 256, 128, -1.25f, 0.5f, stream));
        cases.emplace_back("register_tile_three_tails_alpha_beta", product_case(129, 137, 47, 0.75f, -0.5f, stream));
        cases.emplace_back("register_multiple_tiles_k_tail", product_case(192, 256, 131, -1.25f, 0.0f, stream));
        cases.emplace_back("register_row_dispatch_boundary", product_case(127, 129, 33, 0.5f, 0.25f, stream));
        cases.emplace_back("register_k_dispatch_boundary", product_case(129, 129, 31, 0.5f, 0.0f, stream));
        cases.emplace_back("zero_k_scales_prior", scale_case(true, -2.0f, stream));
        cases.emplace_back("alpha_zero_skips_inputs", scale_case(false, 0.5f, stream));
        cases.emplace_back("zero_k_beta_zero_nan_prior", scale_case(true, 0.0f, stream));
        Buffer aliased(16);
        cases.emplace_back("output_input_alias_rejected", sfGEMM::gemm(aliased.data, aliased.data, aliased.data, 4, 4, 4) == cudaErrorInvalidValue);
        require(cudaStreamDestroy(stream)); stream = nullptr;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        cases.emplace_back("cuda_execution", false);
        if (stream) cudaStreamDestroy(stream);
    }
    bool passed = true;
    for (const auto& entry : cases) passed &= entry.second;
    std::printf("{\"package\":\"sfGEMM\",\"status\":\"%s\",\"cases\":[", passed ? "PASS" : "FAIL");
    for (size_t i = 0; i < cases.size(); ++i)
        std::printf("%s{\"name\":\"%s\",\"status\":\"%s\"}", i ? "," : "", cases[i].first.c_str(), cases[i].second ? "PASS" : "FAIL");
    std::puts("]}");
    return passed ? 0 : 1;
}
