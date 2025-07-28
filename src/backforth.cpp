#include "measurement.hpp"
#include "memory.hpp"
#include "base_kernels.cuh"

__global__ void simple_write_kernel(double *out, size_t n_elems) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n_elems) {
        out[tid] = 0xff;
    }
}

template <typename ALLOC, bool PAGELEVEL>
void backforth_template(size_t n_iter, size_t n_bytes, size_t inner, std::string name) {
    double times[n_iter];

    ALLOC PINGPONG(n_bytes);

    for (size_t iter = 0; iter < n_iter; ++iter) {
        uint64_t start = get_cpu_clock();
        for (size_t i = 0; i < inner; ++i) {
            cudaMemset(PINGPONG.data, 0xff, n_bytes);
        }
        cudaDeviceSynchronize();
        if constexpr (PAGELEVEL) {
            for (size_t j = 0; j < n_bytes; j += PAGE_SIZE) {
                PINGPONG.data[j] = 0xff;
            }
        } else {
            memset(PINGPONG.data, 0xff, n_bytes);
        }
        uint64_t end = get_cpu_clock();
        times[iter] = get_elapsed_milliseconds_clock(start, end);
    }

    std::string subdir;
    if constexpr (PAGELEVEL) {
        subdir = "pagelevel_scalability/";
    } else {
        subdir = "scalability/";
    }

    raw_times_to_file(times, n_iter, "results/apps/transfer/" + subdir + name);
}

template <bool PAGELEVEL>
void run_backforth_tests(size_t n_iter, size_t n_bytes, size_t inner) {
    // backforth_template<HOST_MEM, PAGELEVEL>(n_iter, n_bytes, inner, "/ddr/" + std::to_string(inner));
    // backforth_template<DEVICE_MEM, PAGELEVEL>(n_iter, n_bytes, inner, "/hbm/" + std::to_string(inner));
    // backforth_template<REMOTE_HOST_MEM, PAGELEVEL>(n_iter, n_bytes, inner, "/ddr_remote/" + std::to_string(inner));
    // backforth_template<REMOTE_DEVICE_MEM, PAGELEVEL>(n_iter, n_bytes, inner, "/hbm_remote/" + std::to_string(inner));
    // backforth_template<ManagedMemoryDataFactory, PAGELEVEL>(n_iter, n_bytes, inner, "/managed/" + std::to_string(inner));
    backforth_template<MmapDataFactory, PAGELEVEL>(n_iter, n_bytes, inner, "/mmap/" + std::to_string(inner));
}
