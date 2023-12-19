#include "device_benchmarks.cu"
#include "host_benchmarks.hpp"
#include "constants.hpp"
#include "paradigms/benchmarks.cu"
#include <iostream>
#include <fstream>
#include <sched.h>

int main() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    if (sched_getcpu() != 0) {
        std::cerr << "Setting CPU affinity failed!" << std::endl;
        exit(-1);
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << device_count << " devices found" << std::endl;

    for (size_t i = 32; i < 32 << 10; i <<= 1) {
        // RUN_BENCHMARK(contiguous_host_modified_device_read_benchmark, "results/from_host/" + std::to_string(i), 11, i);
        // RUN_BENCHMARK(contiguous_device_modified_device_read_benchmark, "results/from_device/" + std::to_string(i), 11, i);
        // RUN_BENCHMARK(contiguous_untouched_device_read_benchmark, "results/untouched/" + std::to_string(i), 11, i);
        // RUN_BENCHMARK(contiguous_cuda_malloc_untouched_device_read_benchmark, "results/device_mem/" + std::to_string(i), 11, i);
        // RUN_BENCHMARK(contiguous_cuda_malloc_device_modified_device_read_benchmark, "results/device_initialized_mem/" + std::to_string(i), 11, i);
        RUN_BENCHMARK_RAW(kernel_invocation_benchmark, "results/host/" + std::to_string(i), 101, i, i);
        //RUN_BENCHMARK_RAW(flattened_parallelism_benchmark, "results/flattened/" + std::to_string(i), 101, i, i);
        RUN_BENCHMARK_RAW(dynamic_parallelism_benchmark, "results/dynamic/" + std::to_string(i), 101, i, i);
        RUN_BENCHMARK_RAW(cooperative_parallelism_benchmark, "results/cooperative/" + std::to_string(i), 101, i, i);
    }
    std::cout << std::endl;
}