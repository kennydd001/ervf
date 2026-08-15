#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <vector>
#include <omp.h>

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double at = q * (values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(at));
    const auto hi = static_cast<std::size_t>(std::ceil(at));
    return values[lo] + (values[hi] - values[lo]) * (at - lo);
}

int main() {
    constexpr std::size_t N = 64ULL * 1024ULL * 1024ULL;
    constexpr std::size_t BYTES = N * sizeof(double);
    double *a, *b, *c;
    if (posix_memalign(reinterpret_cast<void**>(&a), 2 * 1024 * 1024, BYTES)
        || posix_memalign(reinterpret_cast<void**>(&b), 2 * 1024 * 1024, BYTES)
        || posix_memalign(reinterpret_cast<void**>(&c), 2 * 1024 * 1024, BYTES)) return 2;
    const int madvise_result = madvise(a, BYTES, MADV_HUGEPAGE) | madvise(b, BYTES, MADV_HUGEPAGE) | madvise(c, BYTES, MADV_HUGEPAGE);
    const int mlock_result = mlock(a, BYTES) | mlock(b, BYTES) | mlock(c, BYTES);
    if (mlock_result == 0) { munlock(a, BYTES); munlock(b, BYTES); munlock(c, BYTES); }
    omp_set_num_threads(16);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    const int thread_counts[] = {1, 4, 8, 16};
    std::cout << std::setprecision(17) << "{\n  \"array_bytes_each\": " << BYTES
              << ",\n  \"madvise_hugepage_success\": " << (madvise_result == 0 ? "true" : "false")
              << ",\n  \"mlock_full_arrays_success\": " << (mlock_result == 0 ? "true" : "false")
              << ",\n  \"threads\": {\n";
    for (int ti = 0; ti < 4; ++ti) {
        const int threads = thread_counts[ti]; omp_set_num_threads(threads);
        std::vector<double> copy, scale, add, triad;
        for (int iteration = 0; iteration < 7; ++iteration) {
            auto begin = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < N; ++i) c[i] = a[i];
            auto end = std::chrono::steady_clock::now();
            copy.push_back((2.0 * BYTES) / std::chrono::duration<double>(end - begin).count() / 1e9);
            begin = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < N; ++i) b[i] = 3.0 * c[i];
            end = std::chrono::steady_clock::now();
            scale.push_back((2.0 * BYTES) / std::chrono::duration<double>(end - begin).count() / 1e9);
            begin = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < N; ++i) c[i] = a[i] + b[i];
            end = std::chrono::steady_clock::now();
            add.push_back((3.0 * BYTES) / std::chrono::duration<double>(end - begin).count() / 1e9);
            begin = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < N; ++i) a[i] = b[i] + 3.0 * c[i];
            end = std::chrono::steady_clock::now();
            triad.push_back((3.0 * BYTES) / std::chrono::duration<double>(end - begin).count() / 1e9);
        }
        std::cout << "    \"" << threads << "\": {"
                  << "\"copy_p50_GBps\": " << percentile(copy, .5) << ", "
                  << "\"copy_p95_GBps\": " << percentile(copy, .95) << ", "
                  << "\"scale_p50_GBps\": " << percentile(scale, .5) << ", "
                  << "\"add_p50_GBps\": " << percentile(add, .5) << ", "
                  << "\"triad_p50_GBps\": " << percentile(triad, .5) << "}"
                  << (ti == 3 ? "\n" : ",\n");
    }
    double checksum = 0.0;
    for (std::size_t i = 0; i < N; i += 4096) checksum += a[i] + b[i] + c[i];
    std::cout << "  },\n  \"checksum\": " << checksum << "\n}\n";
    free(a); free(b); free(c);
    return std::isfinite(checksum) ? 0 : 3;
}
