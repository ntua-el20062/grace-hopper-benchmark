#include "benchmarks.cu"
#include "constants.hpp"
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

    for (size_t i = 1024 * 2; i < CPU_L3_CACHE; i *= 2) {
        RUN_BENCHMARK(contiguous_host_modified_device_write_benchmark, "results/host_modified/" + std::to_string(i), 11, i);
        RUN_BENCHMARK(contiguous_device_modified_device_write_benchmark, "results/device_modified/" + std::to_string(i), 11, i);
        RUN_BENCHMARK(contiguous_untouched_device_write_benchmark, "results/untouched/" + std::to_string(i), 11, i);
    }
    std::cout << std::endl;
}