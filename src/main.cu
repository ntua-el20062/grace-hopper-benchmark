#include "device_benchmarks.cu"
#include "hmm_benchmarks.cu"
#include "host_benchmarks.hpp"
#include "constants.hpp"
#include "paradigms/benchmarks.cu"
#include "atomic_benchmarks.cuh"
#include "thread_tools.hpp"
#include <iostream>
#include <fstream>
#include <sched.h>
#include "stream.hpp"
#include "read_write_copy.hpp"
#include "latency.hpp"

void run_launch_overhead_benchmarks() {
    for (size_t i = 32; i < 32 << 12; i <<= 1) {
        RUN_BENCHMARK_RAW(kernel_invocation_benchmark, "results/host/" + std::to_string(i), 101, i, i);
        RUN_BENCHMARK_RAW(flattened_parallelism_benchmark, "results/flattened/" + std::to_string(i), 101, i, i);
        RUN_BENCHMARK_RAW(dynamic_parallelism_benchmark, "results/dynamic/" + std::to_string(i), 101, i, i);
        RUN_BENCHMARK_RAW(cooperative_parallelism_benchmark, "results/cooperative/" + std::to_string(i), 101, i, i);
    }
}

// void run_managed_memory_benchmarks() {
//     for (size_t i = 2048; i < 1UL << 32; i <<= 1) {
//         RUN_BENCHMARK_THROUGHPUT(contiguous_host_modified_device_read_benchmark, "results/mm/from_host/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(contiguous_device_modified_device_read_benchmark, "results/mm/from_device/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(contiguous_untouched_device_read_benchmark, "results/mm/untouched/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(contiguous_cuda_malloc_untouched_device_read_benchmark, "results/mm/device_mem/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(contiguous_cuda_malloc_device_modified_device_read_benchmark, "results/mm/device_initialized_mem/" + std::to_string(i), 11, i);
//     }
// }

// void run_hmm_benchmarks() {
//     for (size_t i = 64; i < 1UL << 32; i <<= 1) {
//         RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_device_modified_device_write_benchmark, "results/hmm/device_modified/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_modified_device_write_benchmark, "results/hmm/host_modified/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_invalidated_device_write_benchmark, "results/hmm/host_invalidated/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_exclusive_device_write_benchmark, "results/hmm/host_exclusive/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_untouched_device_write_benchmark, "results/hmm/untouched/" + std::to_string(i), 11, i);
//         RUN_BENCHMARK_THROUGHPUT(contiguous_cuda_malloc_untouched_device_write_benchmark, "results/hmm/device_mem/" + std::to_string(i), 11, i);
//         std::cout << i << std::endl;
//     }
// }

void run_tiny_benchmarks() {
    for (size_t i = 4096; i < CPU_L3_CACHE << 2; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 64) * 64;
        RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_untouched_device_write_single_thread_benchmark, "results/tiny/single/untouched/" + std::to_string(i), 101, i);
        RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_modified_device_write_single_thread_benchmark, "results/tiny/single/host_modified/" + std::to_string(i), 101, i);
        RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_device_modified_device_write_single_thread_benchmark, "results/tiny/single/device_modified/" + std::to_string(i), 101, i);
        std::cout << i << std::endl;
    }
    // for (size_t i = 64 * 32; i < CPU_L3_CACHE << 4; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64) * 64;
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_untouched_device_write_single_warp_benchmark, "results/tiny/warp/untouched/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_modified_device_write_single_warp_benchmark, "results/tiny/warp/host_modified/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_device_modified_device_write_single_warp_benchmark, "results/tiny/warp/device_modified/" + std::to_string(i), 101, i);
    //     std::cout << i << std::endl;
    // }
    // for (size_t i = 64 * 1024; i < CPU_L3_CACHE << 4; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64) * 64;
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_untouched_device_write_single_block_benchmark, "results/tiny/block/untouched/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_modified_device_write_single_block_benchmark, "results/tiny/block/host_modified/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_device_modified_device_write_single_block_benchmark, "results/tiny/block/device_modified/" + std::to_string(i), 101, i);
    //     std::cout << i << std::endl;
    // }
    // for (size_t i = 64 * (1024 * 264); i < CPU_L3_CACHE << 12; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64) * 64;
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_untouched_device_write_full_benchmark, "results/tiny/full/untouched/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_modified_device_write_full_benchmark, "results/tiny/full/host_modified/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_device_modified_device_write_full_benchmark, "results/tiny/full/device_modified/" + std::to_string(i), 101, i);
    //     std::cout << i << std::endl;
    // }
}

void run_host_write_benchmarks() {
    for (size_t i = 4096; i < CPU_L3_CACHE << 2; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 64 * 16) * 64 * 16;
        std::cout << i << std::endl;
        RUN_BENCHMARK_LATENCY(contiguous_invalid_host_write_benchmark, "results/host/write/invalid/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_modified_host_write_benchmark<0>, "results/host/write/local_modified/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_modified_host_write_benchmark<1>, "results/host/write/other_modified/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        //RUN_BENCHMARK_LATENCY(contiguous_host_modified_host_write_benchmark<15>, "results/host/write/far_modified/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_shared_host_write_benchmark<1>, "results/host/write/other_shared/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        //RUN_BENCHMARK_LATENCY(contiguous_host_shared_host_write_benchmark<15>, "results/host/write/far_shared/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_exclusive_host_write_benchmark<1>, "results/host/write/other_exclusive/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
    }
}

void run_host_read_benchmarks() {
    for (size_t i = 4096; i < CPU_L3_CACHE << 2; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 64 * 16) * 64 * 16;
        std::cout << i << std::endl;
        RUN_BENCHMARK_LATENCY(contiguous_invalid_host_read_benchmark, "results/host/read/invalid/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_modified_host_read_benchmark<0>, "results/host/read/local_modified/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_modified_host_read_benchmark<1>, "results/host/read/other_modified/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        //RUN_BENCHMARK_LATENCY(contiguous_host_modified_host_read_benchmark<15>, "results/host/read/far_modified/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_shared_host_read_benchmark<1>, "results/host/read/other_shared/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        //RUN_BENCHMARK_LATENCY(contiguous_host_shared_host_read_benchmark<15>, "results/host/read/far_shared/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
        RUN_BENCHMARK_LATENCY(contiguous_host_exclusive_host_read_benchmark<1>, "results/host/read/other_exclusive/" + std::to_string(i), 1000, ((1./1000000.)*(i/64)), i);
    }
}

void print_device_props() {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    std::cout << "Device " << 0 << " - " << deviceProperties.name << ":\n";
    std::cout << "  Compute capability: " << deviceProperties.major << "." << deviceProperties.minor << "\n";
    std::cout << "  Total global memory: " << deviceProperties.totalGlobalMem << " bytes\n";
    std::cout << "  Number of streaming multiprocessors: " << deviceProperties.multiProcessorCount << "\n";
    std::cout << "  Max threads per block: " << deviceProperties.maxThreadsPerBlock << "\n";
    std::cout << "  Max threads per SM: " << deviceProperties.maxThreadsPerMultiProcessor << "\n";
    std::cout << "  Max block dimensions: (" << deviceProperties.maxThreadsDim[0] << ", "
                << deviceProperties.maxThreadsDim[1] << ", " << deviceProperties.maxThreadsDim[2] << ")\n";
    std::cout << "  Max grid dimensions: (" << deviceProperties.maxGridSize[0] << ", "
                << deviceProperties.maxGridSize[1] << ", " << deviceProperties.maxGridSize[2] << ")\n";
    std::cout << "  Warp size: " << deviceProperties.warpSize << "\n";
    std::cout << "  Memory pitch: " << deviceProperties.memPitch << "\n";
    std::cout << "  Max shared memory per block: " << deviceProperties.sharedMemPerBlock << " bytes\n";
    std::cout << "  Total constant memory: " << deviceProperties.totalConstMem << " bytes\n";
    std::cout << "  Clock rate: " << deviceProperties.clockRate << " kHz\n";
    std::cout << "  Texture alignment: " << deviceProperties.textureAlignment << "\n";
    std::cout << "  Device overlap: " << (deviceProperties.deviceOverlap ? "Yes" : "No") << "\n";
    std::cout << "  Integrated GPU: " << (deviceProperties.integrated ? "Yes" : "No") << "\n";
    std::cout << "  Concurrent kernels: " << (deviceProperties.concurrentKernels ? "Yes" : "No") << "\n";
    std::cout << "  ECC enabled: " << (deviceProperties.ECCEnabled ? "Yes" : "No") << "\n";
    std::cout << "  Memory clock rate: " << deviceProperties.memoryClockRate << " kHz\n";
    std::cout << "  Memory bus width: " << deviceProperties.memoryBusWidth << " bits\n";
    std::cout << "  L2 cache size: " << deviceProperties.l2CacheSize << " bytes\n";
    std::cout << "  Max threads per warp: " << deviceProperties.maxThreadsPerMultiProcessor << "\n";
    std::cout << "\n";
}

int main() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpuset), &cpuset);

    if (sched_getcpu() != 0) {
        std::cerr << "Setting CPU affinity failed!" << std::endl;
        exit(-1);
    } else {
        std::cout << "[INFO] thread pinning works" << std::endl;
    }
    // OLD OPENMP CODE
    // //std::cout << "[INFO] default omp threads: " << omp_get_max_threads() << std::endl;
    // omp_set_num_threads(64);
    // std::cout << "[INFO] running omp with threads: " << omp_get_max_threads() << std::endl;
    // #pragma omp parallel
    // {
    //     cpu_set_t mask;
    //     CPU_ZERO(&mask);
    //     sched_getaffinity(0, sizeof(mask), &mask);
    //     for (int i = 0; i < CPU_SETSIZE; ++i) {
    //         if (CPU_ISSET(i, &mask)) {
    //             printf("(%d %d)", omp_get_thread_num(), i);
    //             break;
    //         }
    //     }
    // }
    // printf("\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << device_count << " devices found" << std::endl;

    std::cout << "[INFO] cache fill buffer size: " << sizeof(cache_filler) << std::endl;

    //RUN_BENCHMARK_RAW(host_ping_device_pong_benchmark, "results/atomic/host_ping_device_pong", 1000);
    //RUN_BENCHMARK_RAW(host_ping_host_pong_benchmark, "results/atomic/host_ping_host_pong", 1000);
    
    sleep_test();

    init_thread_array();
    // run_stream_benchmark_device(1UL << 32, 5, 1);
    for (size_t i = 4096; i <= 1UL << 31; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 64) * 64;
        std::cout << i << std::endl;
        // run_read_tests_device(10, i, 0, "", std::to_string(i));
        // run_write_tests_device(10, i, 0, "", std::to_string(i));
        // run_read_tests_host(10, i, 64, "", std::to_string(i));
        // run_write_tests_host(10, i, 64, "", std::to_string(i));
        // run_copy_tests_device(10, i, 264, "", std::to_string(i));
        // run_copy_tests_host(10, i, 64, "", std::to_string(i));
    }
    // run_latency_test_host<true>(1000, 1UL << 31);
    // run_latency_test_device<true>(1000, 1UL << 31);
    // run_latency_test_host<false>(1000, 33554432);
    // run_latency_test_device<false>(1000, 33554432);
    // run_stream_benchmark_host(5, 1UL << 32);


    // ------------- HOST SCALABILITY -------------
    for (size_t n_threads = 1; n_threads <= 64; ++n_threads) {
        run_read_tests_host(10, 1UL << 31, n_threads, "scalability/", std::to_string(n_threads));
        run_write_tests_host(10, 1UL << 31, n_threads, "scalability/", std::to_string(n_threads));
        // run_copy_tests_host(10, 1UL << 31, n_threads, "scalability/", std::to_string(n_threads));
    }

    // ------------- DEVICE SCALABILITY -------------
    for (size_t n_blocks = 1; n_blocks <= 264; ++n_blocks) {
        run_read_tests_device(10, 1UL << 31, n_blocks, "scalability/", std::to_string(n_blocks));
        run_write_tests_device(10, 1UL << 31, n_blocks, "scalability/", std::to_string(n_blocks));
        // run_copy_tests_device(10, 1UL << 31, n_blocks, "scalability/", std::to_string(n_blocks));
    }

    terminate_threads();
}