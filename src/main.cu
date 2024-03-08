#include "device_benchmarks.cu"
#include "hmm_benchmarks.cu"
#include "host_benchmarks.hpp"
#include "constants.hpp"
// clude "paradigms/benchmarks.cu"
#include "atomic_benchmarks.cuh"
#include "thread_tools.hpp"
#include <iostream>
#include <fstream>
#include <sched.h>
#include "stream.hpp"
#include "read_write_copy.hpp"
#include "latency.hpp"
//#include "apps.hpp"
#include "cache.hpp"
#include "sweep.hpp"

// void run_launch_overhead_benchmarks() {
//     for (size_t i = 32; i < 32 << 12; i <<= 1) {
//         RUN_BENCHMARK_RAW(kernel_invocation_benchmark, "results/host/" + std::to_string(i), 101, i, i);
//         RUN_BENCHMARK_RAW(flattened_parallelism_benchmark, "results/flattened/" + std::to_string(i), 101, i, i);
//         // RUN_BENCHMARK_RAW(dynamic_parallelism_benchmark, "results/dynamic/" + std::to_string(i), 101, i, i);
//         RUN_BENCHMARK_RAW(cooperative_parallelism_benchmark, "results/cooperative/" + std::to_string(i), 101, i, i);
//     }
// }

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
    // for (size_t i = 4096; i < CPU_L3_CACHE << 2; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64) * 64;
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_untouched_device_write_single_thread_benchmark, "results/tiny/single/untouched/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_host_modified_device_write_single_thread_benchmark, "results/tiny/single/host_modified/" + std::to_string(i), 101, i);
    //     RUN_BENCHMARK_THROUGHPUT(hmm::contiguous_device_modified_device_write_single_thread_benchmark, "results/tiny/single/device_modified/" + std::to_string(i), 101, i);
    //     std::cout << i << std::endl;
    // }
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

__attribute__((always_inline)) inline uint64_t get_cyclecount() {
    uint64_t value;
    asm volatile("mrs %0, PMCCNTR_EL0" : "=r"(value));  
    return value;
}

int main() {
#ifndef OPENBLAS
    int main_thread = 287; // 287
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(main_thread, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpuset), &cpuset);

    struct sched_param param;
    param.sched_priority = 1;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

    if (sched_getcpu() != main_thread) {
        std::cerr << "Setting CPU affinity failed!" << std::endl;
        exit(-1);
    } else {
        std::cout << "[INFO] thread pinning works" << std::endl;
    }
#endif

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    printf("[INFO] %d devices found\n", device_count);
    printf("[INFO] cache fill buffer size: %lu\n", cache_filler);
    printf("[INFO] system page size: %lu (using %lu)\n", sysconf(_SC_PAGESIZE), PAGE_SIZE);
    std::cout << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        for (int j = 0; j < device_count; ++j) {
            if (i == j) continue;
            cudaDeviceEnablePeerAccess(j, 0);
        }
    }
    cudaSetDevice(0);

    // host_device_clock_test();

    // print_device_props();

    //RUN_BENCHMARK_RAW(host_ping_device_pong_benchmark, "results/atomic/host_ping_device_pong", 1000);
    //RUN_BENCHMARK_RAW(host_ping_host_pong_benchmark, "results/atomic/host_ping_host_pong", 1000);
    
    // sleep_test();

    // run_ping_pong_benchmarks<HOST_MEM>(100);
    // run_ping_pong_benchmarks<DEVICE_MEM>(100);

    init_thread_array();

    // for (size_t n_threads = 5; n_threads <= 70; n_threads += 5) {
    //     for (double ratio = 0.1; ratio <= 0.95; ratio += 0.1) {
    //         std::cout << n_threads << " " << ratio << std::endl;
    //         run_host_sweep_tests(100, 1UL << 32, ratio, n_threads);
    //     }
            // run_host_sweep_tests(100, 1UL << 32, 1.0, n_threads);
    // }

    // for (size_t n_blocks = 0; n_blocks <= 260; n_blocks += 20) {
    //     for (double ratio = 0.01; ratio <= 0.095; ratio += 0.01) {
    //         std::cout << n_blocks << " " << ratio << std::endl;
    //         run_device_sweep_tests(10, 1UL << 32, n_blocks, ratio);
    //     }
    //     for (double ratio = 0.1; ratio <= 0.95; ratio += 0.1) {
    //         std::cout << n_blocks << " " << ratio << std::endl;
    //         run_device_sweep_tests(10, 1UL << 32, n_blocks, ratio);
    //     }
    //     run_device_sweep_tests(10, 1UL << 32, n_blocks, 0.0);
    //     run_device_sweep_tests(10, 1UL << 32, n_blocks, 1.0);
    // }
    // for (double ratio = 0.01; ratio <= 0.095; ratio += 0.01) {
    //     std::cout << 264 << " " << ratio << std::endl;
    //     run_device_sweep_tests(10, 1UL << 32, 264, ratio);
    // }
    // for (double ratio = 0.1; ratio <= 0.95; ratio += 0.1) {
    //     std::cout << 264 << " " << ratio << std::endl;
    //     run_device_sweep_tests(10, 1UL << 32, 264, ratio);
    // }
    // run_device_sweep_tests(10, 1UL << 32, 264, 0.0);
    // run_device_sweep_tests(10, 1UL << 32, 264, 1.0);

    // run_clock_offset_test(72, 0);

    // run_clock_analysis_host();

    // gpu_clock_test();

    // test_cache();

    // run_cuda_memcpy_heatmap_tests();
    // run_device_copy_heatmap_tests();
    // run_host_copy_heatmap_tests();

    // run_stream_benchmark_device(1UL << 32, 5, 1);
    for (size_t i = 4096; i <= 1UL << 33; i = (size_t)((double) i * sqrt(2))) {
        i = CEIL(i, 64 * 72) * 64 * 72;
        // std::cout << i << std::endl;
        // run_thrust_reduction_benchmarks(10, i);
        // run_read_tests_device(10, i, 264, "", std::to_string(i));
        // run_write_tests_device(10, i, 264, "", std::to_string(i));
        // run_copy_tests_device(10, i, 264, "", std::to_string(i));
        // run_read_tests_host(10, i, 72, "", std::to_string(i));
        // run_write_tests_host(10, i, 72, "", std::to_string(i));
        // run_read_tests_device(10, i, 0, "", std::to_string(i));
        // run_write_tests_device(10, i, 0, "", std::to_string(i));
        // run_copy_tests_device(10, i, 0, "", std::to_string(i));
        // run_read_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_write_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_copy_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_memset_tests_host(10, i, 72, "", std::to_string(i));
        // run_memcpy_tests_host(10, i, 72, "", std::to_string(i));
        // run_memset_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_memcpy_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_cuda_memcpy_tests(10, i, "", std::to_string(i));
    }

    for (size_t i = 4096; i <= 1UL << 33; i = (size_t)((double) i * sqrt(2))) {
        i = CEIL(i, 64) * 64;
        // std::cout << i << std::endl;
        // run_thrust_reduction_benchmarks(10, i);
        // run_read_tests_device(10, i, 264, "", std::to_string(i));
        // run_write_tests_device(10, i, 264, "", std::to_string(i));
        // run_copy_tests_device(10, i, 264, "", std::to_string(i));
        // run_read_tests_host(10, i, 72, "", std::to_string(i));
        // run_write_tests_host(10, i, 72, "", std::to_string(i));
        // run_read_tests_device(10, i, 0, "", std::to_string(i));
        // run_write_tests_device(10, i, 0, "", std::to_string(i));
        // run_copy_tests_device(10, i, 0, "", std::to_string(i));
        // run_read_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_write_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_copy_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_memset_tests_host(10, i, 72, "", std::to_string(i));
        // run_memcpy_tests_host(10, i, 72, "", std::to_string(i));
        // run_memset_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_memcpy_tests_host(10, i, 1, "single/", std::to_string(i));
        // run_cuda_memcpy_tests(10, i, "", std::to_string(i));
    }
    // run_read_tests_device(10, 1UL << 32, 264, "", std::to_string(1UL << 32));
    // run_write_tests_device(10, 1UL << 32, 264, "", std::to_string(1UL << 32));
    // run_copy_tests_device(10, 1UL << 32, 264, "", std::to_string(1UL << 32));
    // run_read_tests_device(10, 1UL << 32, 0, "", std::to_string(1UL << 32));
    // run_write_tests_device(10, 1UL << 32, 0, "", std::to_string(1UL << 32));
    // run_copy_tests_device(10, 1UL << 32, 0, "", std::to_string(1UL << 32));
    // run_read_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    // run_write_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    // run_copy_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    // run_read_tests_host(10, 1UL << 32, 72, "", std::to_string(1UL << 32));
    // run_write_tests_host(10, 1UL << 32, 72, "", std::to_string(1UL << 32));
    // run_copy_tests_host(10, 1UL << 32, 72, "", std::to_string(1UL << 32));
    // run_copy_tests_host(10, 1UL << 32, 1, "", std::to_string(1UL << 32));
    // run_stream_benchmark_host(5, 1UL << 32);
    // run_memset_tests_host(10, 1UL << 32, 72, "", std::to_string(1UL << 32));
    // run_memcpy_tests_host(10, 1UL << 32, 72, "", std::to_string(1UL << 32));
    // run_memset_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    // run_memcpy_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    // run_thrust_reduction_benchmarks(10, 1UL << 32);

    // for (size_t i = 4096; i <= 1UL << 32; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64 * 64) * 64 * 64;
    //     std::cout << i << std::endl;
    //     run_read_tests_host(10, i, 64, "", std::to_string(i));
    //     run_write_tests_host(10, i, 64, "", std::to_string(i));
    //     run_copy_tests_host(10, i, 64, "", std::to_string(i));
    // }

    // for (size_t i = 1024 * 4 * 8; i <= 1UL << 20; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 1024 * 4 * 8) * 1024 * 4 * 8;
    //     std::cout << i << std::endl;
    //     // run_read_tests_device(10, i, 264, "", std::to_string(i));
    //     run_read_tests_device(10, i, 1, "", std::to_string(i));
    // }

    run_latency_test_host<true>(100, 1UL << 32); std::cout << "done" << std::endl;
    run_latency_test_device<true>(100, 1UL << 32); std::cout << "done" << std::endl;
    // run_latency_test_host<false>(100, 1UL << 32); std::cout << "done" << std::endl;
    // run_latency_test_device<false>(100, 1UL << 32); std::cout << "done" << std::endl;

    // ------------- LATENCY -------------
    for (size_t i = 6186319744; i <= 1UL << 35; i = (size_t)((double) i * sqrt(2))) {
        // if (i == 6186319744) continue;
        i = CEIL(i, 64) * 64;
        // std::cout << i << std::endl;
        // latency_test_device_template<false, false, HOST_MEM>(10, i, "results/latency/device/scalability/ddr/" + std::to_string(i));
        // latency_test_device_template<false, false, DEVICE_MEM>(10, i, "results/latency/device/scalability/hbm/" + std::to_string(i));
        // latency_test_host_template<false, false, HOST_MEM>(10, i, 0, "results/latency/host/scalability/ddr/" + std::to_string(i));
        // latency_test_host_template<false, false, DEVICE_MEM>(10, i, 0, "results/latency/host/scalability/hbm/" + std::to_string(i));
        // latency_test_device_template<false, false, REMOTE_HOST_MEM>(10, i, "results/latency/device/scalability/ddr_remote/" + std::to_string(i));
        // latency_test_device_template<false, false, REMOTE_DEVICE_MEM>(10, i, "results/latency/device/scalability/hbm_remote/" + std::to_string(i));
        // latency_test_host_template<false, false, REMOTE_HOST_MEM>(10, i, 0, "results/latency/host/scalability/ddr_remote/" + std::to_string(i));
        // latency_test_host_template<false, false, REMOTE_DEVICE_MEM>(10, i, 0, "results/latency/host/scalability/hbm_remote/" + std::to_string(i));

        // latency_test_device_template<false, true, HOST_MEM>(10, i, "results/latency/device/scalability/write/ddr/" + std::to_string(i));
        // latency_test_device_template<false, true, DEVICE_MEM>(10, i, "results/latency/device/scalability/write/hbm/" + std::to_string(i));
        // latency_test_host_template<false, true, HOST_MEM>(10, i, "results/latency/host/scalability/write/ddr/" + std::to_string(i));
        // latency_test_host_template<false, true, DEVICE_MEM>(10, i, "results/latency/host/scalability/write/hbm/" + std::to_string(i));
        // latency_test_device_template<false, true, REMOTE_HOST_MEM>(10, i, "results/latency/device/scalability/write/ddr_remote/" + std::to_string(i));
        // latency_test_device_template<false, true, REMOTE_DEVICE_MEM>(10, i, "results/latency/device/scalability/write/hbm_remote/" + std::to_string(i));
        // latency_test_host_template<false, true, REMOTE_HOST_MEM>(10, i, "results/latency/host/scalability/write/ddr_remote/" + std::to_string(i));
        // latency_test_host_template<false, true, REMOTE_DEVICE_MEM>(10, i, "results/latency/host/scalability/write/hbm_remote/" + std::to_string(i));
    }

    // // ------------- HOST SCALABILITY -------------
    // for (size_t n_threads = 1; n_threads <= 64; ++n_threads) {
    //     std::cout << n_threads << std::endl;
    //     // run_read_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads));
    //     // run_write_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads));
    //     // run_copy_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads));
    //     run_memset_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads));
    //     run_memcpy_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads));
    // }

    // // // ------------- DEVICE SCALABILITY -------------
    // for (size_t n_blocks = 80; n_blocks <= 264; ++n_blocks) {
    //     std::cout << n_blocks << std::endl;
    //     run_read_tests_device(10, 1UL << 33, n_blocks, "scalability/", std::to_string(n_blocks));
    //     run_write_tests_device(10, 1UL << 33, n_blocks, "scalability/", std::to_string(n_blocks));
    // //     run_copy_tests_device(10, 1UL << 33, n_blocks, "scalability/", std::to_string(n_blocks));
    // }

    // // ------------- DEVICE THROUGHPUT -------------
    // for (size_t i = 4096; i <= 1UL << 33; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 1024 * sizeof(double)) * 1024 * sizeof(double);
    //     std::cout << i << std::endl;
    //     run_write_throughput_test_device(10, i);
    //     run_copy_throughput_test_device(10, i);
    // }
    // run_write_throughput_test_device(10, 1UL << 33);
    // run_copy_throughput_test_device(10, 1UL << 33);

    // ------------- DEVICE VERY LARGE  -------------
    // for (size_t i = 10511205312; i >= 1000000000; i = (size_t)((double) i / sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64) * 64;
    //     std::cout << i << std::endl;
    //     if (i == 10511205312) continue;
    //     run_read_tests_device(10, i, 264, "large/", std::to_string(i));
    //     run_write_tests_device(10, i, 264, "large/", std::to_string(i));
    //     run_copy_tests_device(10, i, 264, "large/", std::to_string(i));
    // }

    // init_cublas();

    // for (size_t i = 4096; i <= 1UL << 32; i = (size_t)((double) i * sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64) * 64;
    //     size_t n_elems = i / sizeof(float);
    //     size_t elems_per_side = std::sqrt(n_elems);
    //     i = elems_per_side * elems_per_side * sizeof(float);
    //     std::cout << i << std::endl;
        // run_cublas_gemm_tests(100, i);
        // run_openblas_gemm_tests(10, i);
    // }
    // run_cublas_gemm_tests(10, 1UL << 32);
    // run_openblas_gemm_tests(10, 1UL << 32);

    terminate_threads();
}