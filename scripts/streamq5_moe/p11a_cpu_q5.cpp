#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>

constexpr int EXPERTS = 8;
constexpr int HIDDEN = 2048;
constexpr int INTERMEDIATE = 768;
constexpr std::size_t EXPERT_BYTES = 3035136;
constexpr std::size_t MATRIX_BYTES = 1011712;
constexpr std::size_t CODE_BYTES = 983040;

inline float bf16_to_float(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    return std::bit_cast<float>(bits);
}

inline std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    std::uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7FFFU + lsb) >> 16);
}

inline float round_bf16(float value) { return bf16_to_float(float_to_bf16(value)); }

float exact_dot(const float* x, const unsigned char* matrix, int row, int cols) {
    const unsigned char* packed = matrix + 64;
    const auto* scales = reinterpret_cast<const std::uint16_t*>(matrix + 64 + CODE_BYTES);
    const int packs = cols >> 3;
    const int groups = cols >> 7;
    std::array<float, 256> partial{};
    for (int tid = 0; tid < 256; ++tid) {
        float sum = 0.0f;
        for (int pack = tid; pack < packs; pack += 256) {
            const unsigned char* source = packed + (static_cast<std::int64_t>(row) * packs + pack) * 5;
            const std::uint64_t word = static_cast<std::uint64_t>(source[0])
                | (static_cast<std::uint64_t>(source[1]) << 8)
                | (static_cast<std::uint64_t>(source[2]) << 16)
                | (static_cast<std::uint64_t>(source[3]) << 24)
                | (static_cast<std::uint64_t>(source[4]) << 32);
            const int column = pack << 3;
            const float scale = bf16_to_float(scales[row * groups + (column >> 7)]);
            for (int item = 0; item < 8; ++item) {
                const int code = static_cast<int>((word >> (item * 5)) & 31ULL) - 15;
                const float weight = round_bf16(static_cast<float>(code) * scale);
                sum += weight * x[column + item];
            }
        }
        partial[tid] = sum;
    }
    for (int stride = 128; stride > 0; stride >>= 1)
        for (int tid = 0; tid < stride; ++tid) partial[tid] += partial[tid + stride];
    return round_bf16(partial[0]);
}

struct Runtime {
    std::vector<unsigned char> bank;
    std::vector<float> x, gate, up, down;

    explicit Runtime(const std::string& path)
        : bank(EXPERTS * EXPERT_BYTES), x(HIDDEN), gate(EXPERTS * INTERMEDIATE),
          up(EXPERTS * INTERMEDIATE), down(EXPERTS * HIDDEN) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open Q5 layer");
        input.read(reinterpret_cast<char*>(bank.data()), bank.size());
        if (input.gcount() != static_cast<std::streamsize>(bank.size())) throw std::runtime_error("short bank read");
        for (int column = 0; column < HIDDEN; ++column)
            x[column] = static_cast<float>((column * 17) % 257 - 128) / 64.0f;
    }

    void run() {
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < EXPERTS * INTERMEDIATE; ++index) {
            const int expert = index / INTERMEDIATE;
            const int row = index % INTERMEDIATE;
            const unsigned char* base = bank.data() + expert * EXPERT_BYTES;
            gate[index] = exact_dot(x.data(), base, row, HIDDEN);
            up[index] = exact_dot(x.data(), base + MATRIX_BYTES, row, HIDDEN);
        }
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < EXPERTS * INTERMEDIATE; ++index) {
            const float value = gate[index];
            const float silu = round_bf16(value / (1.0f + std::exp(-value)));
            gate[index] = round_bf16(silu * up[index]);
        }
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < EXPERTS * HIDDEN; ++index) {
            const int expert = index / HIDDEN;
            const int row = index % HIDDEN;
            const unsigned char* base = bank.data() + expert * EXPERT_BYTES + 2 * MATRIX_BYTES;
            down[index] = exact_dot(gate.data() + expert * INTERMEDIATE, base, row, INTERMEDIATE);
        }
    }
};

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double at = q * (values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(at));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(at));
    return values[lo] + (values[hi] - values[lo]) * (at - lo);
}

std::vector<double> measure(Runtime& runtime, int threads, int iterations) {
    omp_set_num_threads(threads);
    runtime.run();
    std::vector<double> values;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto begin = std::chrono::steady_clock::now();
        runtime.run();
        const auto end = std::chrono::steady_clock::now();
        values.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return values;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: p11a_cpu_q5 LAYER OUTPUT\n"; return 2; }
    Runtime runtime(argv[1]);
    const std::array<int, 4> threads{1, 4, 8, 16};
    std::array<std::vector<double>, 4> validation;
    int selected = 0;
    for (int index = 0; index < 4; ++index) {
        validation[index] = measure(runtime, threads[index], 3);
        if (percentile(validation[index], 0.5) < percentile(validation[selected], 0.5)) selected = index;
    }
    auto test = measure(runtime, threads[selected], 20);
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(runtime.gate.data()), runtime.gate.size() * sizeof(float));
    output.write(reinterpret_cast<const char*>(runtime.up.data()), runtime.up.size() * sizeof(float));
    output.write(reinterpret_cast<const char*>(runtime.down.data()), runtime.down.size() * sizeof(float));
    output.close();
    bool finite = true;
    for (float value : runtime.gate) finite = finite && std::isfinite(value);
    for (float value : runtime.up) finite = finite && std::isfinite(value);
    for (float value : runtime.down) finite = finite && std::isfinite(value);
    std::cout << std::setprecision(17) << "{\n  \"validation\": {\n";
    for (int index = 0; index < 4; ++index) {
        std::cout << "    \"" << threads[index] << "\": {\"values_ms\": [";
        for (std::size_t i = 0; i < validation[index].size(); ++i) std::cout << (i ? ", " : "") << validation[index][i];
        std::cout << "], \"p50_ms\": " << percentile(validation[index], .5) << "}" << (index == 3 ? "\n" : ",\n");
    }
    std::cout << "  },\n  \"selected_threads\": " << threads[selected] << ",\n  \"test_ms\": [";
    for (std::size_t i = 0; i < test.size(); ++i) std::cout << (i ? ", " : "") << test[i];
    std::cout << "],\n  \"test_p50_ms\": " << percentile(test, .5)
              << ",\n  \"test_p95_ms\": " << percentile(test, .95)
              << ",\n  \"finite\": " << (finite ? "true" : "false") << "\n}\n";
    return finite ? 0 : 3;
}
