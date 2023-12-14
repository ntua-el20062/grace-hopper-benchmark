#include "benchmarks.cu"
#include <iostream>
#include <fstream>
#include <sched.h>

#define RUN_BENCHMARK(FUNCNAME, NITER, ...) {\
    std::ofstream file("results/" #FUNCNAME ".csv");\
    for (size_t i = 0; i < NITER; ++i) {\
        file << FUNCNAME(__VA_ARGS__) << std::endl;\
    }}

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

    RUN_BENCHMARK(contiguous_untouched_float_device_read_benchmark, 11, PAGE_SIZE << 1);
}