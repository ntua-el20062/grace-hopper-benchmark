#include "device_benchmarks.cu"
#include "hmm_benchmarks.cu"
#include "host_benchmarks.hpp"
#include "constants.hpp"
//#include "paradigms/benchmarks.cu"
#include "atomic_benchmarks.cuh"
#include "thread_tools.hpp"
#include <iostream>
#include <fstream>
#include <sched.h>
#include "stream.hpp"
#include "read_write_copy.hpp"
#include "latency.hpp"
#include "apps.hpp"
#include "cache.hpp"
#include "sweep.hpp"
#include "backforth.cpp"
#include "pm_counters.hpp"
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>


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

void print_power() {
    pm_counters power;
    sleep(0.1);
    power.start();
    sleep(1);
    power.stop();
    power.print_result();
}

void segfault_handler(int sig) {
    void *array[20];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 20);

    fprintf(stderr, "Error: signal %d (Segmentation fault):\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}


int main() {
    signal(SIGSEGV, segfault_handler);

#ifndef OPENBLAS
    int main_thread = 71;
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
       std::cout << "[INFO] thread pinning works: CORE " << main_thread << std::endl;
    }
#else
    #pragma omp parallel
    {
        printf("%d\n", sched_getcpu());
    }
#endif
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    printf("[INFO] %d devices found\n", device_count);
    printf("[INFO] num threads: %lu\n", NUM_THREADS);
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
     

    host_device_clock_test();

    print_device_props();
    
    sleep_test();
    run_ping_pong_benchmarks<HOST_MEM>(100);
    //run_ping_pong_benchmarks<DEVICE_MEM>(100);
    //std::cout << "DONE WITH PING PONG BENCHMARKS"<< std::endl;
    
    // TO THELW AYTO ---------------------------------------------------------------
    //run_cuda_memcpy_heatmap_tests();

    /* for (size_t i = 1; i <= 2048; i *= 2) {
         std::cout << i << std::endl;
         run_backforth_tests<true>(10, 1UL << 30, i);
     }*/

    print_power();
    
    //init_cublas();
    /*   
    for (size_t i = 4096; i <= 1UL << 32; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 64) * 64;
        size_t n_elems = i / sizeof(float);
        size_t elems_per_side = std::sqrt(n_elems);
        i = elems_per_side * elems_per_side * sizeof(float);
        std::cout << i << std::endl;
	int num_of_iter = 100;
        run_cublas_gemm_tests<double>((size_t)(num_of_iter), i);
	std::cout << "RUN CUBLAS TESTS"<< std::endl;
    }
    
    for (size_t i = 4096; i <= 1UL << 33; i = (size_t)((double) i * std::sqrt(2))) {
        run_cublas_gemm_tests<double>(static_cast<size_t>(10), i);
        run_cublas_gemm_tests<TF32>(static_cast<size_t>(10), i);
        run_cublas_gemm_tests<float>(static_cast<size_t>(10), i);
        run_cublas_gemm_tests<__half>(static_cast<size_t>(10), i);
	        std::cout << "RUN CUBLAS 2 TESTS"<< std::endl;

    }
   */ 

    init_thread_array(main_thread);

    std::string base = "";

    // base = "/noisy/";
    // size_t noise_bytes = CEIL(1UL << 33, 64 * 72) * 64 * 72;
    // DEVICE_MEM infinite(noise_bytes);
    // run_test(INFINITE_READ, 72, 0, infinite.data, nullptr, (noise_bytes) / sizeof(double), nullptr, true);

    // size_t noise_bytes = 1UL << 33;
    // HOST_MEM infinite(noise_bytes);
    // clock_t dummy = 0;
    // infinite_device_read_kernel<<<1024, 264>>>((ulonglong2 *)infinite.data, noise_bytes / sizeof(double), &dummy);
    // printf("NOISE STARTED!\n");

    // print_power();

    //return;


    
    run_clock_offset_test(72, 0); 
    
    run_clock_analysis_host();

    //gpu_clock_test();

    test_cache();
    
    //std::cout << "BEFORE HOST  HEAT MAP"<< std::endl;

    //run_device_copy_heatmap_tests();
    run_host_copy_heatmap_tests();
    
    for (size_t i = 4096; i <= 1UL << 33; i = (size_t)((double) i * sqrt(2))) {
        i = CEIL(i, 64 * 72) * 64 * 72;
        std::cout << i << std::endl;
        //run_read_tests_host(10, i, 72, "72/", std::to_string(i));
        run_write_tests_host(10, i, 72, "72/", std::to_string(i));
	//run_copy_tests_host(10, i, 72, "72/", std::to_string(i));
        //run_read_tests_host(10, i, 1, "single/", std::to_string(i));
        run_write_tests_host(10, i, 1, "single/", std::to_string(i));
        //run_copy_tests_host(10, i, 1, "single/", std::to_string(i));
        //run_memset_tests_host(10, i, 72, "72/", std::to_string(i));
        //run_memcpy_tests_host(10, i, 72, "72/", std::to_string(i));
        //run_memset_tests_host(10, i, 1, "single/", std::to_string(i));
        //run_memcpy_tests_host(10, i, 1, "single/", std::to_string(i));
    }
    /*
    for (size_t i = 4096; i <= 1UL << 33; i = (size_t)((double) i * sqrt(2))) {
        i = CEIL(i, 64) * 64;
        std::cout << i << std::endl;
        run_thrust_reduction_benchmarks(10, i);
	std::cout << "done with thrust reduction benchmarks 1" << std::endl;
        run_read_tests_device(10, i, 264, base, std::to_string(i));
        run_write_tests_device(10, i, 264, base, std::to_string(i));
        run_copy_tests_device(10, i, 264, base, std::to_string(i));
        run_thrust_reduction_benchmarks(10, i);
	std::cout << "done with thrust reduction benchmarks 2" << std::endl;

        run_read_tests_device(10, i, 264, "", std::to_string(i));
        run_write_tests_device(10, i, 264, "", std::to_string(i));
        run_copy_tests_device(10, i, 264, "", std::to_string(i));
        run_read_tests_device(10, i, 0, "", std::to_string(i));
	std::cout << "done with read device" << std::endl;

        run_write_tests_device(10, i, 0, "", std::to_string(i));
	std::cout << "done with write device" << std::endl;

        run_copy_tests_device(10, i, 0, "", std::to_string(i));
	std::cout << "done with copy device" << std::endl;
        //run_cuda_memcpy_tests(10, i, "", std::to_string(i));
    }
    */

    //run_stream_benchmark_device(1UL << 32, 5, 1);


    /*
    
    //run_read_tests_device(100, 1UL << 33, 264, base, std::to_string(1UL << 33));
    //run_write_tests_device(100, 1UL << 33, 264, base, std::to_string(1UL << 33));
    //std::cout << "1" << std::endl;
    //run_copy_tests_device(10, 1UL << 32, 264, base, std::to_string(1UL << 32));
    //std::cout << "2" << std::endl;
    
    run_read_tests_device(2, 1UL << 33, 1, "block/", std::to_string(1UL << 33));
    std::cout << "3" << std::endl;

    run_write_tests_device(2, 1UL << 33, 1, "block/", std::to_string(1UL << 33));
    std::cout << "4" << std::endl;

    run_copy_tests_device(10, 1UL << 32, 0, "", std::to_string(1UL << 32));
    std::cout << "5" << std::endl;
    */
  
    //run_read_tests_host(100, 1UL << 33, 1, "single/", std::to_string(1UL << 33));
    std::cout << "6" << std::endl;

    run_write_tests_host(100, 1UL << 33, 1, "single/", std::to_string(1UL << 33));
    std::cout << "7" << std::endl;

    //run_copy_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    std::cout << "8" << std::endl;

    //run_read_tests_host(100, 1UL << 33, 72, "72/", std::to_string(1UL << 33));
    std::cout << "9" << std::endl;

    run_write_tests_host(100, 1UL << 33, 72, "72/", std::to_string(1UL << 33));
    std::cout << "10" << std::endl;

    //run_copy_tests_host(10, 1UL << 32, 72, "72/", std::to_string(1UL << 32));
    std::cout << "11" << std::endl;

    //run_copy_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    std::cout << "12" << std::endl;

    //run_stream_benchmark_host(5, 1UL << 32);
    std::cout << "13" << std::endl;

    //run_memset_tests_host(10, 1UL << 32, 72, "72/", std::to_string(1UL << 32));
    std::cout << "14" << std::endl;

    //run_memcpy_tests_host(10, 1UL << 32, 72, "72/", std::to_string(1UL << 32));
    std::cout << "15" << std::endl;

    //run_memset_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    std::cout << "16" << std::endl;

    //run_memcpy_tests_host(10, 1UL << 32, 1, "single/", std::to_string(1UL << 32));
    std::cout << "17" << std::endl;
    /*
       run_thrust_reduction_benchmarks(10, 1UL << 32);
       std::cout << "18 and now the for loop" << std::endl;
    */
    
    /* for (size_t i = 4096; i <= 1UL << 32; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 64 * 64) * 64 * 64;
        std::cout << i << std::endl;
        run_read_tests_host(10, i, 64, "64/", std::to_string(i));
        run_write_tests_host(10, i, 64, "64/", std::to_string(i));
        run_copy_tests_host(10, i, 64, "64/", std::to_string(i));
    }*/
    /*
    for (size_t i = 1024 * 4 * 8; i <= 1UL << 20; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 1024 * 4 * 8) * 1024 * 4 * 8;
        std::cout << i << std::endl;
        //run_read_tests_device(10, i, 264, "", std::to_string(i));
        run_read_tests_device(10, i, 1, "", std::to_string(i));
    }
    */  

    //run_latency_test_host<false>(2, 1UL << 34); std::cout << "done latency test host" << std::endl;
   
    //run_latency_test_device<false>(100, 1UL << 32); std::cout << "done latency test device" << std::endl;
    
    // ------------- LATENCY -------------
    /*
    for (size_t i = 4096; i <= 1UL << 35; i = (size_t)((double) i * sqrt(2))) {
        i = CEIL(i, 64) * 64;
        std::cout << i << std::endl;
        latency_test_device_template<false, false, HOST_MEM>(10, i, "results/latency/device/scalability/ddr/" + std::to_string(i));    //-----------------------------------------------TO THELW
        latency_test_device_template<false, false, DEVICE_MEM>(10, i,  "results/latency/device/scalability/hbm/" + std::to_string(i)); //--------------------------------------------TO THELW
        latency_test_host_template<false, false, HOST_MEM>(10, i, 0, "results/latency/host/scalability/ddr/" + std::to_string(i));       //---------------------------------------------TO THELW
        latency_test_host_template<false, false, DEVICE_MEM>(10, i, 0, "results/latency/host/scalability/hbm/" + std::to_string(i));     //---------------------------------------------TO THELW
          //latency_test_device_template<false, false, REMOTE_HOST_MEM>(10, i, "results/latency/device/scalability/ddr_remote/" + std::to_string(i));
          //latency_test_device_template<false, false, REMOTE_DEVICE_MEM>(10, i, "results/latency/device/scalability/hbm_remote/" + std::to_string(i));
          //latency_test_host_template<false, false, REMOTE_HOST_MEM>(10, i, 0, "results/latency/host/scalability/ddr_remote/" + std::to_string(i));
          //latency_test_host_template<false, false, REMOTE_DEVICE_MEM>(10, i, 0, "results/latency/host/scalability/hbm_remote/" + std::to_string(i));
        
        latency_test_device_template<false, true, HOST_MEM>(10, i,  "results/latency/device/scalability/write/ddr/" + std::to_string(i));   //--------------------------------------------TO THELW
        latency_test_device_template<false, true, DEVICE_MEM>(10, i, "results/latency/device/scalability/write/hbm/" + std::to_string(i))   //--------------------------------------------TO THELW
        latency_test_host_template<false, true, HOST_MEM>(10, i, 0, "results/latency/host/scalability/write/ddr/" + std::to_string(i));       //--------------------------------------------TO THELW
        latency_test_host_template<false, true, DEVICE_MEM>(10, i, 0, "results/latency/host/scalability/write/hbm/" + std::to_string(i));     //--------------------------------------------TO THELW
          //latency_test_device_template<false, true, REMOTE_HOST_MEM>(10, i, "results/latency/device/scalability/write/ddr_remote/" + std::to_string(i));
          //latency_test_device_template<false, true, REMOTE_DEVICE_MEM>(10, i, "results/latency/device/scalability/write/hbm_remote/" + std::to_string(i));
          //latency_test_host_template<false, true, REMOTE_HOST_MEM>(10, i, "results/latency/host/scalability/write/ddr_remote/" + std::to_string(i));
          //latency_test_host_template<false, true, REMOTE_DEVICE_MEM>(10, i, "results/latency/host/scalability/write/hbm_remote/" + std::to_string(i));
    }*/
    // ------------- HOST SCALABILITY -------------
    for (size_t n_threads = 1; n_threads <= 72; ++n_threads) {
        std::cout << n_threads << std::endl;
        //run_read_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads)); //-----------------------------------------------------------------------------------------TO THELW
        run_write_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads));  //-----------------------------------------------------------------------------------------TO THELW
        //run_copy_tests_host(10, 1UL << 33, n_threads, "scalability/", std::to_string(n_threads));
    }

    /*------------------------------------------------------------------------------------ta thelw ola ta apo katw---------------------------------------------------------------------------------------
    // ------------- DEVICE SCALABILITY -------------
    for (size_t n_blocks = 1; n_blocks <= 264; ++n_blocks) {
        std::cout << n_blocks << std::endl;
        run_read_tests_device(10, 1UL << 33, n_blocks, "scalability/", std::to_string(n_blocks));
        run_write_tests_device(10, 1UL << 33, n_blocks, "scalability/", std::to_string(n_blocks));
        run_copy_tests_device(10, 1UL << 33, n_blocks, "scalability/", std::to_string(n_blocks));
    }

    // // ------------- DEVICE THROUGHPUT -------------
    for (size_t i = 4096; i <= 1UL << 33; i = (size_t)((double) i * sqrt(sqrt(2)))) {
        i = CEIL(i, 1024 * sizeof(double)) * 1024 * sizeof(double);
        std::cout << i << std::endl;
        run_write_throughput_test_device(10, i);
        run_copy_throughput_test_device(10, i);
    }
    run_write_throughput_test_device(10, 1UL << 33);
    run_copy_throughput_test_device(10, 1UL << 33);
    // ------------- DEVICE VERY LARGE  -------------
    // for (size_t i = 10511205312; i >= 1000000000; i = (size_t)((double) i / sqrt(sqrt(2)))) {
    //     i = CEIL(i, 64) * 64;
    //     std::cout << i << std::endl;
    //     if (i == 10511205312) continue;
    //     run_read_tests_device(10, i, 264, "large/", std::to_string(i));
    //     run_write_tests_device(10, i, 264, "large/", std::to_string(i));
    //     run_copy_tests_device(10, i, 264, "large/", std::to_string(i));
    // }

    */
    terminate_threads();
}
