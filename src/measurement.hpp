#pragma once

#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <omp.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// write in GB/s
void times_to_file(clock_t *times, size_t n_iterations, size_t n_bytes, std::string path) {
    std::ofstream file(path);
    const double freq = 1980000000.;
    for (size_t i = 0; i < n_iterations; ++i) {
        double elapsed = (double) times[i] / freq;
        // double elapsed = times[i] / 1000.;
        file << (double) n_bytes /  (elapsed * 1000000000.) << std::endl;
    }
}

void millisecond_times_to_gb_sec_file(double *times, size_t n_iterations, size_t n_bytes, std::string path) {
    std::ofstream file(path);
    for (size_t i = 0; i < n_iterations; ++i) {
        double elapsed = times[i] / 1000.;
        file << (double) n_bytes /  (elapsed * 1000000000.) << std::endl;
    }
}

void millisecond_times_to_latency_ns_file(double *times, size_t n_iterations, size_t n_elems, std::string path) {
    std::ofstream file(path);
    for (size_t i = 0; i < n_iterations; ++i) {
        double elapsed_ns = times[i] * 1000000.;
        file << elapsed_ns / n_elems << std::endl;
    }
}

#define RUN_BENCHMARK_THROUGHPUT(FUNCNAME, OUTNAME, NITER, BYTES) {\
    double measurements[NITER];\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        measurements[__i] = ((double) BYTES / (1000. * 1000.)) / FUNCNAME(BYTES);\
    }\
    std::ofstream file(OUTNAME);\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        file << measurements[__i] << std::endl;\
    }}

#define RUN_BENCHMARK_RAW(FUNCNAME, OUTNAME, NITER, ...) {\
    double measurements[NITER];\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        measurements[__i] = FUNCNAME(__VA_ARGS__);\
    }\
    std::ofstream file(OUTNAME);\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        file << measurements[__i] << std::endl;\
    }}

#define RUN_BENCHMARK_LATENCY(FUNCNAME, OUTNAME, NITER, ELEMS, ...) {\
    double measurements[NITER];\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        measurements[__i] = FUNCNAME(__VA_ARGS__) / (double) ELEMS;\
    }\
    std::ofstream file(OUTNAME);\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        file << measurements[__i] << std::endl;\
    }}

double time_kernel_execution(const void *kernel, int grid_size, int block_size, void **args, size_t shared_memory, cudaStream_t stream) {
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    cudaLaunchKernel(kernel, grid_size, block_size, args, shared_memory, stream);
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double) time;
}

int get_gpu_clock_khz() {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    return deviceProperties.clockRate;
}

double time_kernel_execution_clock(const void *kernel, int grid_size, int block_size, void **args, size_t n_args, size_t times_size, size_t shared_memory, cudaStream_t stream) {
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

    double time = (double) (max_stop - min_start) / ((double) get_gpu_clock_khz());

    cudaFree(device_start);
    cudaFree(device_stop);
    free(start);
    free(stop);

    return time;
}

double time_cooperative_kernel_execution(const void *kernel, void **args, size_t shared_memory, cudaStream_t stream) {
    float time;
    cudaEvent_t start, stop;

    int grid_size, block_size;

    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    cudaLaunchCooperativeKernel(kernel, grid_size, block_size, args, shared_memory, stream);
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double) time;
}

double get_elapsed_milliseconds(struct timeval start, struct timeval end) {
    return (double) (end.tv_usec - start.tv_usec) / 1000. + (double) (end.tv_sec - start.tv_sec) * 1000.;
}

__attribute__((always_inline)) inline uint64_t get_cpu_freq() {
    uint64_t freq;

    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));

    return freq;
}

__attribute__((always_inline)) inline uint64_t get_cpu_clock() {
    uint64_t tsc;

    asm volatile("isb" : : : "memory");
    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc));

    return tsc;
}

double clock_to_milliseconds(uint64_t c) {
    double freq = (double) get_cpu_freq();

    return ((double)(c)) / (freq / 1000.);

}

double get_elapsed_milliseconds_clock(uint64_t start, uint64_t end) {
    return clock_to_milliseconds(end - start);
}

void clock_granularity_test() {
    std::ofstream file("results/clock_granularity");
    for (size_t i = 0; i < 1000000; ++i) {
        uint64_t start_clock = get_cpu_clock();
        uint64_t end_clock = get_cpu_clock();
        file << end_clock - start_clock << std::endl;
    }
}

__global__ void clock_granularity_kernel(clock_t *out) {
    for (size_t i = 0; i < 1000000; ++i) {
        clock_t start_clock = clock();
        clock_t end_clock = clock();
        out[i] = end_clock - start_clock;
    }
}

void device_clock_granularity_test() {
    std::ofstream file("results/device_clock_granularity");
    clock_t *times = (clock_t *) malloc(1000000 * sizeof(clock_t));
    clock_granularity_kernel<<<1, 1>>>(times);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < 1000000; ++i) {
        file << times[i] << std::endl;
    }
    free(times);
}


void sleep_test() {
    clock_granularity_test();
    device_clock_granularity_test();

    uint64_t start_clock = get_cpu_clock();
    uint64_t end_clock = get_cpu_clock();
    std::cout << "[INFO] overhead of cycles: " << end_clock - start_clock << std::endl;

    start_clock = get_cpu_clock();
    sleep(1);
    end_clock = get_cpu_clock();

    std::cout << "[INFO] CPU timer runs at " << (double) get_cpu_freq() / 1000000. << "MHz" << std::endl;
    std::cout << "[INFO] GPU timer runs at " << (double) get_gpu_clock_khz() / 1000. << "MHz" << std::endl;
    std::cout << "[INFO] target 1000.0, elapsed: " << get_elapsed_milliseconds_clock(start_clock, end_clock) << std::endl;
    
    start_clock = get_cpu_clock();
    asm volatile("mov x0, #10000;"
                 "mov x1, #0;"
                 "_EMPTY_LOOP_TAG:;"
                 "add     x1, x1, #0x1;"
                 "cmp     x1, x0;"
                 "b.ne    _EMPTY_LOOP_TAG;" ::: "x0", "x1");
    end_clock = get_cpu_clock();

    std::cout << "[INFO] loop iteration overhead: " << get_elapsed_milliseconds_clock(start_clock, end_clock) * 100. << "ns (" << (double) (end_clock - start_clock) / 10000 << " cycles)" << std::endl;
}

#define TIME_FUNCTION_EXECUTION(TIME, FUNC, ...) {\
    uint64_t __start = get_cpu_clock();\
    FUNC(__VA_ARGS__);\
    uint64_t __end = get_cpu_clock();\
    TIME = get_elapsed_milliseconds_clock(__start, __end);}

template <typename FUNCTYPE, typename... ARGTYPES>
double time_function_execution(FUNCTYPE f, ARGTYPES... args) {
    uint64_t start = get_cpu_clock();
    f(args...);
    uint64_t end = get_cpu_clock();

    return get_elapsed_milliseconds_clock(start, end);
}

#define GENERIC_SYNC(__TARGET, ID, FUNC, DELAY)\
    uint64_t __CHECK;\
    if (ID == 0) {\
        __CHECK = FUNC() + DELAY;\
        *__TARGET = __CHECK;\
    } else {\
        do {\
            __CHECK = *__TARGET;\
        } while (__CHECK == 0);\
    }\
    while (FUNC() < __CHECK);

#define KERNEL_SYNC(__TARGET) GENERIC_SYNC(__TARGET, (threadIdx.x + blockDim.x * blockIdx.x), clock, ((1980000/7)*7));
#define OMP_SYNC(__TARGET) GENERIC_SYNC(__TARGET, (omp_get_thread_num()), get_cpu_clock, ((1000000/32)*32));

#define GENERIC_MEASURE(__TARGET, ID, FUNC)\
    auto __TIME = FUNC();\
    __TARGET[ID] = __TIME - __CHECK;

#define KERNEL_MEASURE(__TARGET) GENERIC_MEASURE(__TARGET, (threadIdx.x + blockDim.x * blockIdx.x), clock)
#define OMP_MEASURE(__TARGET) GENERIC_MEASURE(__TARGET, (omp_get_thread_num()), get_cpu_clock)

#ifdef _OPENMP
#define MEASURE_CPU_LOOP_AND_RETURN(LOOP) {\
    volatile uint64_t TIMES[n_threads];\
    TIMES[0] = 0;\
_Pragma("omp parallel num_threads(n_threads)")\
    {\
        OMP_SYNC(TIMES);\
        _Pragma("omp for")\
        LOOP\
        OMP_MEASURE(TIMES);\
    }\
    volatile uint64_t OUT_TIME = TIMES[0];\
    for (size_t i = 0; i < n_threads; ++i) {\
        OUT_TIME = std::max(OUT_TIME, TIMES[i]);\
    }\
    return clock_to_milliseconds(OUT_TIME);}
#else
#define MEASURE_CPU_LOOP_AND_RETURN(LOOP) {\
    auto START = get_cpu_clock();\
    LOOP\
    auto OUT_TIME = get_cpu_clock() - START;\
    return clock_to_milliseconds(OUT_TIME);}
#endif
