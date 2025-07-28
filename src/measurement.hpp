#pragma once

#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <filesystem>
#define CEIL(a, b) (((a)+(b)-1)/(b))

// write in GB/s
void times_to_file(clock_t *times, size_t n_iterations, size_t n_bytes, std::string path, double freq = 1980000000.) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

	std::ofstream file(path);
    for (size_t i = 0; i < n_iterations; ++i) {
        double elapsed = (double) times[i] / freq;
        // double elapsed = times[i] / 1000.;
        file << (double) n_bytes /  (elapsed * 1000000000.) << std::endl;
    }
}

/*
  Writes throughput values (GB/s) computed from raw GPU clock cycle counts (clock_t).
  times is an array of clock cycle counts for each iteration.
  n_iterations is number of timing samples.
  n_bytes is the number of bytes processed in each iteration.
  freq is GPU clock frequency in Hz (default 1.98 GHz).

  For each iteration:
	Convert clock cycles to seconds: elapsed = cycles / freq.
	Calculate throughput: n_bytes / (elapsed * 10^9) converts bytes/sec to GB/s.
	Write throughput to output file line-by-line.

  Each element in this array (times[i]) holds a measured number of GPU clock cycles for one iteration of some operation (usually a kernel execution or a timed section).
  clock_t is a type used to represent clock cycles or timer ticks — in this context, typically GPU clock cycles.

  n_iterations:This represents the number of timing measurements stored in the times array.
*/

void millisecond_times_to_gb_sec_file(double *times, size_t n_iterations, size_t n_bytes, std::string path) { //similar to above, but here times are milliseconds instead of clock cycles, to store in GB/s
std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    	std::ofstream file(path);
    for (size_t i = 0; i < n_iterations; ++i) {
        double elapsed = times[i] / 1000.;
        file << (double) n_bytes /  (elapsed * 1000000000.) << std::endl;
    }
}

void raw_times_to_file(double *times, size_t n_iterations, std::string path) { //no conversion
std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    	std::ofstream file(path);
    for (size_t i = 0; i < n_iterations; ++i) {
        file << times[i] << std::endl;
    }
}

void millisecond_times_to_gb_sec_file(float *times, size_t n_iterations, size_t n_bytes, std::string path) { //float=32bit, double=64bit, maybe floats for cuda timings
std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    	std::ofstream file(path);
    for (size_t i = 0; i < n_iterations; ++i) {
        float elapsed = times[i] / 1000.;
        file << (float) n_bytes /  (elapsed * 1000000000.) << std::endl;
    }
}

void millisecond_times_to_latency_ns_file(double *times, size_t n_iterations, size_t n_elems, std::string path) {
std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    	std::ofstream file(path);
    for (size_t i = 0; i < n_iterations; ++i) {
        double elapsed_ns = times[i] * 1000000.; //from milli to nanosec
        file << elapsed_ns / n_elems << std::endl; //this is latency per element
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

/*
Macro to benchmark throughput.
Runs FUNCNAME(BYTES) NITER times, stores result in measurements.
FUNCNAME(BYTES) is assumed to return execution time in milliseconds.
Throughput = bytes / (time in seconds) = (BYTES / 1,000,000) / time_ms = MB/ms = GB/s.
Writes all throughput measurements to file OUTNAME
 */

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
/*
 * measures latency per element for FUNCNAME
 */

double time_kernel_execution(const void *kernel, int grid_size, int block_size, void **args, size_t shared_memory, cudaStream_t stream) {
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start); //for recording time
    cudaEventCreate(&stop);  //also for recording time

    cudaEventRecord(start, stream); //mark the start of the specified stream, this marks just before the kernel is launched
    cudaLaunchKernel(kernel, grid_size, block_size, args, shared_memory, stream); //launch kernel
    cudaEventRecord(stop, stream); //mark the exact end timestamp of the stream

    cudaEventSynchronize(stop); //block the cpu up until the kernel has finished(stop timestamp)on the gpu
    cudaEventElapsedTime(&time, start, stop); //calculate the time the gpu kernel took and return it

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double) time;
}

int get_gpu_clock_khz() {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    return deviceProperties.clockRate;
}


//measures the execution time of a CUDA kernel using GPU clock counters instead of CUDA events, times_size=possibly the number of threads
double time_kernel_execution_clock(const void *kernel, int grid_size, int block_size, void **args, size_t n_args, size_t times_size, size_t shared_memory, cudaStream_t stream) {
    clock_t *device_start, *device_stop;
    clock_t *start, *stop;

    cudaMalloc(&device_start, sizeof(clock_t) * times_size); //allocate device memory arrays for measurements for start clock  
    cudaMalloc(&device_stop, sizeof(clock_t) * times_size);  //allocate device memory array for stop clock for measurements
    start = (clock_t *) malloc(sizeof(clock_t) * times_size); //allocate array on cpu side for start
    stop = (clock_t *) malloc(sizeof(clock_t) * times_size); //allocate array on cpu side for stop

    void **new_args = (void **) alloca(sizeof(void *) * (n_args + 2)); //new argument list will have all original kernel arguments plus two additional pointers to device start and stop clock arrays
    for (size_t i = 0; i < n_args; ++i) {
        new_args[i] = args[i];
    }
    new_args[n_args] = (void *) &device_start;
    new_args[n_args + 1] = (void *) &device_stop;

    cudaLaunchKernel(kernel, grid_size, block_size, new_args, shared_memory, stream); //launch kernel on specific stream
    cudaDeviceSynchronize(); //wait until kernel finishes for valid data

    cudaMemcpy(start, device_start, sizeof(clock_t) * times_size, cudaMemcpyDeviceToHost); //coppy start and stop arrays to host(arrays from above with malloc)
    cudaMemcpy(stop, device_stop, sizeof(clock_t) * times_size, cudaMemcpyDeviceToHost);

    clock_t min_start = start[0]; //find earliest start and latest stop
    clock_t max_stop = stop[0];
    for (size_t i = 1; i < times_size; ++i) {
        if (start[i] < min_start) {
            min_start = start[i];
        }

        if (stop[i] > max_stop) {
            max_stop = stop[i];
        }
    }

    double time = (double) (max_stop - min_start) / ((double) get_gpu_clock_khz()); //calculate elapsed time

    cudaFree(device_start);
    cudaFree(device_stop);
    free(start);
    free(stop);

    return time;
}

double time_cooperative_kernel_execution(const void *kernel, void **args, size_t shared_memory, cudaStream_t stream) { //time taken for cooperative kernel, no 2 kernel simultaneously, interblock syncronization
    float time;
    cudaEvent_t start, stop;

    int grid_size, block_size;

    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel); //get the best block size for the max occupancy, no idle threads

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream); //mark start of kernel
    cudaLaunchCooperativeKernel(kernel, grid_size, block_size, args, shared_memory, stream); //launch kernel
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double) time;
}

/*
Think of a stream as a command queue for the GPU.

It's more like a timeline or channel through which GPU tasks are issued.
You can have multiple streams, each queuing commands independently.
Commands in the same stream run sequentially in order.
Commands in different streams can run concurrently (overlapping execution) if the hardware supports it.

What does a stream manage?
The GPU hardware executes kernels and memory ops submitted in a stream in the order they appear.
If you submit kernel launches or memory operations to the same stream, they will happen one after the other.
If you use multiple streams, kernels or memory operations in different streams may overlap (run in parallel).

the resources are shared between all streams.
The CPU issues commands into streams, and the GPU executes them asynchronously
*/

double get_elapsed_milliseconds(struct timeval start, struct timeval end) {
    return (double) (end.tv_usec - start.tv_usec) / 1000. + (double) (end.tv_sec - start.tv_sec) * 1000.;
}

__attribute__((always_inline)) inline uint64_t get_cpu_freq() {
    uint64_t freq;

    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq) :: "memory");

    return freq;
}

__attribute__((always_inline)) inline uint64_t get_cpu_clock() { //get cpu clock counter
    uint64_t tsc;

    asm volatile("isb" : : : "memory");
    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc) :: "memory"); // alternative is cntpct_el0

    return tsc;
}

__attribute__((always_inline)) __device__ inline clock_t get_gpu_clock() { //get gpu clock counter
    uint64_t tsc;

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tsc));

    return tsc;
}

double clock_to_milliseconds(uint64_t c) { //Converts a clock count (c) into elapsed time in milliseconds
    double freq = (double) get_cpu_freq();

    return ((double)(c)) / (freq / 1000.);

}

double get_elapsed_milliseconds_clock(uint64_t start, uint64_t end) { //Calculates elapsed milliseconds between two clock counter readings
    return clock_to_milliseconds(end - start);
}

void clock_granularity_test() { //smallest time difference cpu can detect
std::filesystem::create_directories(std::filesystem::path("results/clock_granularity").parent_path());

    	std::ofstream file("results/clock_granularity");
    for (size_t i = 0; i < 10000; ++i) {
        uint64_t start_clock = get_cpu_clock();
        uint64_t end_clock = get_cpu_clock();
        file << end_clock - start_clock << std::endl;
    }
}

__global__ void clock_granularity_kernel(clock_t *out) { //smallest time difference gpu can detect
    for (size_t i = 0; i < 10000; ++i) {
        clock_t start_clock = clock();
        clock_t end_clock = clock();
        out[i] = end_clock - start_clock;
    }
}

__global__ void global_clock_granularity_kernel(clock_t *out) {
    for (size_t i = 0; i < 10000; ++i) {
        // clock_t start_clock = get_gpu_clock();
        // clock_t end_clock = get_gpu_clock();
        // out[i] = end_clock - start_clock;
    }
}

__global__ void basic_loop_overhead_kernel(size_t n_iter, clock_t *measure, size_t *global_dummy) { //understand the cost of a basic loop on a gpu
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
    // *clock = get_gpu_clock();
}

void host_device_clock_test() {
    uint64_t host_clock, device_clock;
    cudaEvent_t e;
    cudaEventCreate(&e);
    // get_clock_kernel<<<1, 1>>>(&device_clock),
    cudaEventSynchronize(e);
    host_clock = get_cpu_clock();
    cudaEventDestroy(e);

    printf("%lu\n%lu\n", host_clock, device_clock);
}

void kernel_loop_overhead_test() {
    clock_t measure;
    size_t global_dummy;
    std::filesystem::create_directories(std::filesystem::path("results/kernel_loop_overhead").parent_path());

    std::ofstream file("results/kernel_loop_overhead");
    for (size_t n_iter = 1; n_iter < 1 << 16; ++n_iter) {
        // basic_loop_overhead_kernel<<<1, 1>>>(n_iter, &measure, &global_dummy);
        cudaDeviceSynchronize();
        file << measure << std::endl;
    }
}

void device_clock_granularity_test() {
    clock_t *times = (clock_t *) malloc(10000 * sizeof(clock_t));
    {
	    std::filesystem::create_directories(std::filesystem::path("results/device_clock_granularity").parent_path());

        std::ofstream file("results/device_clock_granularity");
        // clock_granularity_kernel<<<1, 1>>>(times);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < 10000; ++i) {
            file << times[i] << std::endl;
        }
    }
    {
	    std::filesystem::create_directories(std::filesystem::path("results/device_global_clock_granularity").parent_path());

        std::ofstream file("results/device_global_clock_granularity");
        // global_clock_granularity_kernel<<<1, 1>>>(times);
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


/*

A CUDA event is a marker you can place in a stream to record when certain work starts or finishes. You can use events to measure timing, or to synchronize between CPU and GPU or between streams.

Events can be recorded at any point in a stream.
When you record an event, CUDA captures the GPU timestamp at that point after all previous work in the stream completes.
You can query an event to check if the GPU work before it is done, or wait on it from the CPU or other streams.
Events are lightweight and designed for fine-grained timing and synchronization.

Typical use cases:
Measure elapsed time for a kernel or group of kernels.
Synchronize the host CPU with GPU progress.
Synchronize different streams to ensure one finishes before another starts.

*/


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
// __global__ void gpu_clock_test_kernel(clock_t *global_timesteps, clock_t *local_timesteps) {
//     __shared__ clock_t gt[1024];
//     __shared__ clock_t lt[1024];

//     for (size_t i = 0; i < 1024; ++i) {
//         gt[i] = get_gpu_clock();
//         lt[i] = clock();
//     }

//     for (size_t i = 0; i < 1024; ++i) {
//         global_timesteps[i + blockIdx.x * 1024] = gt[i];
//         local_timesteps[i + blockIdx.x * 1024] = lt[i];
//     }
// }

void gpu_clock_test() {
	std::cout << "started gpu clock test !!!!!!!!!!!!!!!0!" << std::endl;
    clock_t *global_timesteps = (clock_t *) alloca(sizeof(clock_t) * 1024 * 264);
    clock_t *local_timesteps = (clock_t *) alloca(sizeof(clock_t) * 1024 * 264);

    // gpu_clock_test_kernel<<<264, 1>>>(global_timesteps, local_timesteps);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < 264; ++i) { // for each block/file
        std::filesystem::create_directories(std::filesystem::path("results/gpu_clock/global").parent_path());

        std::ofstream global_file("results/gpu_clock/global/" + std::to_string(i));
	std::filesystem::create_directories(std::filesystem::path("results/gpu_clock/local").parent_path());

        std::ofstream local_file("results/gpu_clock/local/" + std::to_string(i));
        for (size_t j = 0; j < 1024; ++j) {
            global_file << global_timesteps[j + i * 1024] << std::endl;
            local_file << local_timesteps[j + i * 1024] << std::endl;
        }
    }
            std::cout << "DONE WITH gpu clock test !!!!!!!!!!!!!!!0!" << std::endl;

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


/*
 Thread 0 sets a synchronization time in the future (current clock + delay).
Other threads wait until thread 0 writes this value.
Then all threads spin until that time is reached.
FUNC:reads current time
 */


#define KERNEL_SYNC(__TARGET) GENERIC_SYNC(__TARGET, (threadIdx.x + blockDim.x * blockIdx.x), clock, ((1980000/7)*7));

#define GENERIC_MEASURE(__TARGET, ID, FUNC)\
    auto __TIME = FUNC();\
    __TARGET[ID] = __TIME - __CHECK;

#define KERNEL_MEASURE(__TARGET) GENERIC_MEASURE(__TARGET, (threadIdx.x + blockDim.x * blockIdx.x), clock)
