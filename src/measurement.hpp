#pragma once

#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define RUN_BENCHMARK_THROUGHPUT(FUNCNAME, OUTNAME, NITER, BYTES) {\
    float measurements[NITER];\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        measurements[__i] = ((float) BYTES / (1000.f * 1000.f)) / FUNCNAME(BYTES);\
    }\
    std::ofstream file(OUTNAME);\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        file << measurements[__i] << std::endl;\
    }}

#define RUN_BENCHMARK_RAW(FUNCNAME, OUTNAME, NITER, ...) {\
    float measurements[NITER];\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        measurements[__i] = FUNCNAME(__VA_ARGS__);\
    }\
    std::ofstream file(OUTNAME);\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        file << measurements[__i] << std::endl;\
    }}

float time_kernel_execution(const void *kernel, int grid_size, int block_size, void **args, size_t shared_memory, cudaStream_t stream) {
    float time;
    cudaEvent_t start, stop;

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaEventRecord(start, stream) );
    gpuErrchk( cudaLaunchKernel(kernel, grid_size, block_size, args, shared_memory, stream) );
    gpuErrchk( cudaEventRecord(stop, stream) );

    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );

    return time;
}

float time_kernel_execution_clock(const void *kernel, int grid_size, int block_size, void **args, size_t n_args, size_t times_size, size_t shared_memory, cudaStream_t stream) {
    clock_t *device_start, *device_stop;
    clock_t *start, *stop;

    cudaMalloc(&device_start, sizeof(clock_t) * times_size);
    cudaMalloc(&device_stop, sizeof(clock_t) * times_size);
    start = (clock_t *) malloc(sizeof(clock_t) * times_size);
    stop = (clock_t *) malloc(sizeof(clock_t) * times_size);

    void **new_args = (void **) alloca(sizeof(void *) * (n_args + 2));
    for (size_t i = 0; i < n_args; ++i) {
        new_args[i] = args[i];
    }
    new_args[n_args] = (void *) &device_start;
    new_args[n_args + 1] = (void *) &device_stop;

    gpuErrchk( cudaLaunchKernel(kernel, grid_size, block_size, new_args, shared_memory, stream) );
    cudaDeviceSynchronize();

    cudaMemcpy(start, device_start, sizeof(clock_t) * times_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(stop, device_stop, sizeof(clock_t) * times_size, cudaMemcpyDeviceToHost);

    clock_t min_start = start[0];
    clock_t max_stop = stop[0];
    for (size_t i = 1; i < times_size; ++i) {
        if (start[i] < min_start) {
            min_start = start[i];
        }

        if (stop[i] > max_stop) {
            max_stop = stop[i];
        }
    }

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    float time = (float) (max_stop - min_start) / ((float) deviceProperties.clockRate);

    cudaFree(device_start);
    cudaFree(device_stop);
    free(start);
    free(stop);

    return time;
}

float time_cooperative_kernel_execution(const void *kernel, void **args, size_t shared_memory, cudaStream_t stream) {
    float time;
    cudaEvent_t start, stop;

    int grid_size, block_size;

    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel);

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaEventRecord(start, stream) );
    gpuErrchk( cudaLaunchCooperativeKernel(kernel, grid_size, block_size, args, shared_memory, stream) );
    gpuErrchk( cudaEventRecord(stop, stream) );

    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );

    return time;
}

float get_elapsed_milliseconds(struct timeval start, struct timeval end) {
    return (float) (end.tv_usec - start.tv_usec) / 1000.f + (float) (end.tv_sec - start.tv_sec) * 1000.f;
}

__always_inline uint64_t get_cpu_freq() {
    uint64_t freq;

    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));

    return freq;
}

__always_inline uint64_t get_cpu_clock() {
    uint64_t tsc;

    asm volatile("isb" : : : "memory");
    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc));

    return tsc;
}

float get_elapsed_milliseconds_clock(uint64_t start, uint64_t end) {
    float freq = (float) get_cpu_freq();

    return ((float)(end - start - 32)) / (freq / 1000.f);
}

void sleep_test() {
    uint64_t oh_start_clock = get_cpu_clock();
    uint64_t oh_end_clock = get_cpu_clock();

    uint64_t start_clock = get_cpu_clock();
    sleep(1);
    uint64_t end_clock = get_cpu_clock();

    std::cout << "target 1000.0, elapsed: " << get_elapsed_milliseconds_clock(start_clock, end_clock) << " with overhead of cycles: " << oh_end_clock - oh_start_clock << std::endl;
}

template <typename FUNCTYPE, typename... ARGTYPES>
float time_function_execution(FUNCTYPE f, ARGTYPES... args) {
    uint64_t start = get_cpu_clock();
    f(args...);
    uint64_t end = get_cpu_clock();

    return get_elapsed_milliseconds_clock(start, end);
}