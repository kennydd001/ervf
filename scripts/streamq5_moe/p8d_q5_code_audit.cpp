#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

namespace fs = std::filesystem;

constexpr std::uint64_t LAYERS = 48;
constexpr std::uint64_t RECORDS_PER_LAYER = 384;
constexpr std::uint64_t RECORD_BYTES = 1011712;
constexpr std::uint64_t HEADER_BYTES = 64;
constexpr std::uint64_t CODE_BYTES = 983040;
constexpr std::uint64_t CODES_PER_RECORD = 1572864;

using Histogram = std::array<std::uint64_t, 32>;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: p8d_q5_code_audit BANK_DIR\n";
        return 2;
    }
    fs::path root(argv[1]);
    Histogram total{};
    std::array<Histogram, 3> projections{};
    std::array<std::uint64_t, LAYERS> layer_overflow{};
    std::array<std::uint64_t, LAYERS> layer_codes{};
    std::uint64_t records = 0;
    const double started = omp_get_wtime();

    for (std::uint64_t layer = 0; layer < LAYERS; ++layer) {
        std::ostringstream filename;
        filename << "layer_" << std::setfill('0') << std::setw(2) << layer << ".q5bin";
        fs::path path = root / filename.str();
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            std::cerr << "cannot open " << path << "\n";
            return 3;
        }
        const auto size = fs::file_size(path);
        if (size != RECORDS_PER_LAYER * RECORD_BYTES) {
            std::cerr << "unexpected size for " << path << ": " << size << "\n";
            return 4;
        }
        std::vector<unsigned char> payload(RECORDS_PER_LAYER * CODE_BYTES);
        for (std::uint64_t record = 0; record < RECORDS_PER_LAYER; ++record) {
            input.seekg(static_cast<std::streamoff>(record * RECORD_BYTES + HEADER_BYTES));
            input.read(reinterpret_cast<char*>(payload.data() + record * CODE_BYTES), CODE_BYTES);
            if (!input) {
                std::cerr << "short read in " << path << " record " << record << "\n";
                return 5;
            }
        }
        const int max_threads = omp_get_max_threads();
        std::vector<Histogram> thread_total(max_threads);
        std::vector<std::array<Histogram, 3>> thread_projection(max_threads);
        std::vector<std::uint64_t> thread_overflow(max_threads, 0);

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& mine = thread_total[tid];
            auto& mine_projection = thread_projection[tid];
            std::uint64_t mine_overflow = 0;
            #pragma omp for schedule(static)
            for (std::int64_t record = 0; record < static_cast<std::int64_t>(RECORDS_PER_LAYER); ++record) {
                const int projection = static_cast<int>(record % 3);
                const unsigned char* data = payload.data() + record * CODE_BYTES;
                for (std::uint64_t offset = 0; offset < CODE_BYTES; offset += 5) {
                    const std::uint64_t word = static_cast<std::uint64_t>(data[offset])
                        | (static_cast<std::uint64_t>(data[offset + 1]) << 8)
                        | (static_cast<std::uint64_t>(data[offset + 2]) << 16)
                        | (static_cast<std::uint64_t>(data[offset + 3]) << 24)
                        | (static_cast<std::uint64_t>(data[offset + 4]) << 32);
                    #pragma unroll
                    for (int item = 0; item < 8; ++item) {
                        const unsigned code = static_cast<unsigned>((word >> (item * 5)) & 31ULL);
                        ++mine[code];
                        ++mine_projection[projection][code];
                        const int signed_code = static_cast<int>(code) - 15;
                        mine_overflow += (signed_code < -7 || signed_code > 7);
                    }
                }
            }
            thread_overflow[tid] = mine_overflow;
        }
        for (int tid = 0; tid < max_threads; ++tid) {
            layer_overflow[layer] += thread_overflow[tid];
            for (int code = 0; code < 32; ++code) {
                total[code] += thread_total[tid][code];
                for (int projection = 0; projection < 3; ++projection)
                    projections[projection][code] += thread_projection[tid][projection][code];
            }
        }
        layer_codes[layer] = RECORDS_PER_LAYER * CODES_PER_RECORD;
        records += RECORDS_PER_LAYER;
        std::cerr << "{\"layer_complete\":" << layer << "}\n";
    }

    std::uint64_t codes = 0;
    std::uint64_t overflow = 0;
    double entropy = 0.0;
    for (int code = 0; code < 32; ++code) {
        codes += total[code];
        const int signed_code = code - 15;
        if (signed_code < -7 || signed_code > 7) overflow += total[code];
    }
    for (int code = 0; code < 32; ++code) {
        if (!total[code]) continue;
        const double probability = static_cast<double>(total[code]) / static_cast<double>(codes);
        entropy -= probability * std::log2(probability);
    }
    const double overflow_fraction = static_cast<double>(overflow) / static_cast<double>(codes);
    const long double original_bits = static_cast<long double>(codes) * 5.0L;
    const long double conservative_bits = static_cast<long double>(codes) * 4.0L
        + static_cast<long double>(overflow) * 37.0L;
    const double conservative_ratio = static_cast<double>(conservative_bits / original_bits);

    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"kind\": \"streamq5_moe_p8d_exact_q5_code_audit\",\n";
    std::cout << "  \"records\": " << records << ",\n";
    std::cout << "  \"codes\": " << codes << ",\n";
    std::cout << "  \"histogram_unsigned_0_31\": [";
    for (int code = 0; code < 32; ++code) std::cout << (code ? ", " : "") << total[code];
    std::cout << "],\n";
    std::cout << "  \"entropy_bits_per_code\": " << entropy << ",\n";
    std::cout << "  \"overflow_abs_gt_7\": " << overflow << ",\n";
    std::cout << "  \"overflow_fraction\": " << overflow_fraction << ",\n";
    std::cout << "  \"conservative_int4_plus_index_value_ratio_to_q5\": " << conservative_ratio << ",\n";
    std::cout << "  \"projection_histograms\": {\n";
    const char* names[] = {"gate", "up", "down"};
    for (int projection = 0; projection < 3; ++projection) {
        std::cout << "    \"" << names[projection] << "\": [";
        for (int code = 0; code < 32; ++code) std::cout << (code ? ", " : "") << projections[projection][code];
        std::cout << "]" << (projection == 2 ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"layer_overflow_abs_gt_7\": [";
    for (std::uint64_t layer = 0; layer < LAYERS; ++layer)
        std::cout << (layer ? ", " : "") << layer_overflow[layer];
    std::cout << "],\n";
    std::cout << "  \"layer_codes\": [";
    for (std::uint64_t layer = 0; layer < LAYERS; ++layer)
        std::cout << (layer ? ", " : "") << layer_codes[layer];
    std::cout << "],\n";
    std::cout << "  \"wall_seconds\": " << (omp_get_wtime() - started) << "\n";
    std::cout << "}\n";
    return codes == 28991029248ULL && records == 18432 ? 0 : 6;
}
