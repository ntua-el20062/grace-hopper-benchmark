#pragma once

#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>

#define CEIL(a, b) (((a)+(b)-1)/(b))

// write in GB/s
void times_to_file(clock_t *times, size_t n_iterations, size_t n_bytes, std::string path, double freq = 1980000000.) {
    std::ofstream file(path);
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

    cudaLaunchKernel(kernel, grid_size, block_size, new_args, shared_memory, stream);
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
    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc)); // alternative is cntpct_el0

    return tsc;
}

__attribute__((always_inline)) __device__ inline clock_t get_gpu_clock() {
    uint64_t tsc;

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tsc));

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
    for (size_t i = 0; i < 10000; ++i) {
        uint64_t start_clock = get_cpu_clock();
        uint64_t end_clock = get_cpu_clock();
        file << end_clock - start_clock << std::endl;
    }
}

__global__ void clock_granularity_kernel(clock_t *out) {
    for (size_t i = 0; i < 10000; ++i) {
        clock_t start_clock = clock();
        clock_t end_clock = clock();
        out[i] = end_clock - start_clock;
    }
}

__global__ void global_clock_granularity_kernel(clock_t *out) {
    for (size_t i = 0; i < 10000; ++i) {
        clock_t start_clock = get_gpu_clock();
        clock_t end_clock = get_gpu_clock();
        out[i] = end_clock - start_clock;
    }
}

__global__ void basic_loop_overhead_kernel(size_t n_iter, clock_t *measure, size_t *global_dummy) {
    size_t dummy;

    clock_t start = clock();

    for (size_t i = 0; i < n_iter; ++i) {
        dummy += i;
    }

    clock_t end = clock();

    *measure = end - start;
    *global_dummy = dummy;
}

__global__ void get_clock_kernel(uint64_t *clock) {
    *clock = get_gpu_clock();
}

void host_device_clock_test() {
    uint64_t host_clock, device_clock;
    cudaEvent_t e;
    cudaEventCreate(&e);
    get_clock_kernel<<<1, 1>>>(&device_clock),
    cudaEventSynchronize(e);
    host_clock = get_cpu_clock();
    cudaEventDestroy(e);

    printf("%lu\n%lu\n", host_clock, device_clock);
}

void kernel_loop_overhead_test() {
    clock_t measure;
    size_t global_dummy;
    std::ofstream file("results/kernel_loop_overhead");
    for (size_t n_iter = 1; n_iter < 1 << 16; ++n_iter) {
        basic_loop_overhead_kernel<<<1, 1>>>(n_iter, &measure, &global_dummy);
        cudaDeviceSynchronize();
        file << measure << std::endl;
    }
}

void device_clock_granularity_test() {
    clock_t *times = (clock_t *) malloc(10000 * sizeof(clock_t));
    {
        std::ofstream file("results/device_clock_granularity");
        clock_granularity_kernel<<<1, 1>>>(times);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < 10000; ++i) {
            file << times[i] << std::endl;
        }
    }
    {
        std::ofstream file("results/device_global_clock_granularity");
        global_clock_granularity_kernel<<<1, 1>>>(times);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < 10000; ++i) {
            file << times[i] << std::endl;
        }
    }
    free(times);
}

__global__ void gpu_sleep(int gpu_freq_khz, double time_ms, clock_t *dummy) {
    clock_t c = clock() + gpu_freq_khz * time_ms;
    while (clock() < c);
    *dummy = c;
}

void thread_clock_function(size_t n_iter, size_t tid) {
    for (size_t i = 0; i < n_iter; ++i) {
        sleep(1);
        printf("%lu:\t%lu\n", tid, get_cpu_clock());
    }
}

void sleep_test() {
    // clock_granularity_test();
    // device_clock_granularity_test();
    // kernel_loop_overhead_test();

    // std::vector<size_t> tids = {1, 72, 144, 216};
    // std::vector<std::thread> threads;
    // for (auto tid : tids) {
    //     threads.emplace_back(thread_clock_function, 10, tid);
    //     cpu_set_t cpuset;
    //     CPU_ZERO(&cpuset);
    //     CPU_SET(0, &cpuset);
    //     pthread_setaffinity_np(threads.back().native_handle(), sizeof(cpuset), &cpuset);
    // }
    // for (auto &t : threads) {
    //     t.join();
    // }

    uint64_t start_clock = get_cpu_clock();
    uint64_t end_clock = get_cpu_clock();
    std::cout << "[INFO] overhead of cycles: " << end_clock - start_clock << std::endl;

    // clock_t dummy;
    // start_clock = get_cpu_clock();
    // gpu_sleep<<<1, 1>>>(get_gpu_clock_khz(), 10000, &dummy);
    // cudaDeviceSynchronize();
    // end_clock = get_cpu_clock();

    std::cout << "[INFO] CPU timer runs at " << (double) get_cpu_freq() / 1000000. << "MHz" << std::endl;
    std::cout << "[INFO] GPU timer runs at " << (double) get_gpu_clock_khz() / 1000. << "MHz" << std::endl;
    // std::cout << "[INFO] target 10000.0, elapsed: " << get_elapsed_milliseconds_clock(start_clock, end_clock) << std::endl;
    
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

// launch with one thread per block
__global__ void gpu_clock_test_kernel(clock_t *global_timesteps, clock_t *local_timesteps) {
    __shared__ clock_t gt[1024];
    __shared__ clock_t lt[1024];

    for (size_t i = 0; i < 1024; ++i) {
        gt[i] = get_gpu_clock();
        lt[i] = clock();
    }

    for (size_t i = 0; i < 1024; ++i) {
        global_timesteps[i + blockIdx.x * 1024] = gt[i];
        local_timesteps[i + blockIdx.x * 1024] = lt[i];
    }
}

void gpu_clock_test() {
    clock_t *global_timesteps = (clock_t *) alloca(sizeof(clock_t) * 1024 * 264);
    clock_t *local_timesteps = (clock_t *) alloca(sizeof(clock_t) * 1024 * 264);

    gpu_clock_test_kernel<<<264, 1>>>(global_timesteps, local_timesteps);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < 264; ++i) { // for each block/file
        std::ofstream global_file("results/gpu_clock/global/" + std::to_string(i));
        std::ofstream local_file("results/gpu_clock/local/" + std::to_string(i));
        for (size_t j = 0; j < 1024; ++j) {
            global_file << global_timesteps[j + i * 1024] << std::endl;
            local_file << local_timesteps[j + i * 1024] << std::endl;
        }
    }
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

#define GENERIC_MEASURE(__TARGET, ID, FUNC)\
    auto __TIME = FUNC();\
    __TARGET[ID] = __TIME - __CHECK;

#define KERNEL_MEASURE(__TARGET) GENERIC_MEASURE(__TARGET, (threadIdx.x + blockDim.x * blockIdx.x), clock)
