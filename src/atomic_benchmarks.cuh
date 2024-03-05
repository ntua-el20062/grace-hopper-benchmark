#pragma once

#include <cuda/std/atomic>
#include "measurement.hpp"

#define TIMES10(a) a a a a a a a a a a
#define TIMES100(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a)
#define TIMES1000(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a)

#define FLAG_A 0
#define FLAG_B 1

extern "C" {
    void host_ping_function(std::atomic<uint8_t> *flag, double *time);
    void host_pong_function(std::atomic<uint8_t> *flag);
}

void host_ping_function_volatile(volatile uint8_t *flag, double *time) {
    while (*flag == FLAG_B);
    uint8_t expected = FLAG_A;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (*flag == FLAG_B);
        *flag = FLAG_B;
    }
    uint64_t end = get_cpu_clock();
    *time = get_elapsed_milliseconds_clock(start, end);
}

void host_pong_function_volatile(volatile uint8_t *flag) {
    uint8_t expected = FLAG_B;
    *flag == FLAG_A;
    for (size_t i = 0; i < 10000; ++i) {
        while (*flag == FLAG_A);
        *flag = FLAG_A;
    }
}

__global__ void device_pong_kernel(cuda::std::atomic<uint8_t> *flag, uint *ret) {
    uint val;
    asm("mov.u32 %0, %smid;" : "=r"(val) );
    *ret = val;
    flag->store(FLAG_A);
    uint8_t expected = FLAG_B;
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, FLAG_A, cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed)) {
            expected = FLAG_B;
        }
    }
}

__global__ void device_ping_kernel(cuda::std::atomic<uint8_t> *flag, clock_t *time, uint *ret) {
    uint val;
    asm("mov.u32 %0, %smid;" : "=r"(val) );
    *ret = val;
    uint8_t expected = FLAG_A;
    while (flag->load() == FLAG_B);

    clock_t start = clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, FLAG_B, cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed)) {
            expected = FLAG_A;
        }
    }
    clock_t end = clock();
    *time = end - start;
}

__global__ void device_pong_kernel_volatile(volatile uint8_t *flag, uint *ret) {
    uint val;
    asm("mov.u32 %0, %smid;" : "=r"(val) );
    *ret = val;
    *flag = FLAG_A;
    for (size_t i = 0; i < 10000; ++i) {
        while (*flag == FLAG_A);
        *flag = FLAG_A;
    }
}

__global__ void device_ping_kernel_volatile(volatile uint8_t *flag, clock_t *time, uint *ret) {
    uint val;
    asm("mov.u32 %0, %smid;" : "=r"(val) );
    *ret = val;
    uint8_t expected = FLAG_A;
    while (*flag == FLAG_B);

    clock_t start = clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (*flag == FLAG_B);
        *flag = FLAG_B;
    }
    clock_t end = clock();
    *time = end - start;
}

double host_ping_device_pong_benchmark(int cpu, int device) {
    MmapDataFactory factory(128);
    cuda::std::atomic<uint8_t> *flag = (cuda::std::atomic<uint8_t> *) factory.data;
    new (flag) cuda::std::atomic<uint8_t>{FLAG_B};

    double time;
    std::thread t(host_ping_function, (std::atomic<uint8_t> *) flag, &time);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);

    uint ret;
    cudaSetDevice(device);
    device_pong_kernel<<<1, 1>>>(flag, &ret);

    t.join();

    cudaDeviceSynchronize();
    cudaSetDevice(0);
    
    return time;
}

double device_ping_host_pong_benchmark(int cpu, int device) {
    MmapDataFactory factory(128);
    cuda::std::atomic<uint8_t> *flag = (cuda::std::atomic<uint8_t> *) factory.data;
    new (flag) cuda::std::atomic<uint8_t>{FLAG_B};

    std::thread t(host_pong_function, (std::atomic<uint8_t> *) flag);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    clock_t time;
    uint ret;
    cudaSetDevice(device);
    device_ping_kernel<<<1, 1>>>(flag, &time, &ret);

    t.join();

    cudaDeviceSynchronize();
    cudaSetDevice(0);
    
    return (double) time / (double) get_gpu_clock_khz();
}

double host_ping_host_pong_benchmark(int cpu_a, int cpu_b) {
    MmapDataFactory factory(128);
    auto *flag = (std::atomic<uint8_t> *) factory.data;
    new (flag) std::atomic<uint8_t>{FLAG_B};

    std::thread t_a(host_pong_function, flag);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_a, &cpuset);
    pthread_setaffinity_np(t_a.native_handle(), sizeof(cpu_set_t), &cpuset);

    double time;
    std::thread t_b(host_ping_function, flag, &time);
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_b, &cpuset);
    pthread_setaffinity_np(t_b.native_handle(), sizeof(cpu_set_t), &cpuset);


    t_a.join();
    t_b.join();

    return time;
}

double device_ping_device_pong_benchmark(int device_a, int device_b) {
    MmapDataFactory factory(128);
    auto *flag = (cuda::std::atomic<uint8_t> *) factory.data;
    new (flag) cuda::std::atomic<uint8_t>{FLAG_B};
    cudaStream_t stream_a, stream_b;
    clock_t time;

    uint ret_ping;
    cudaSetDevice(device_a);
    cudaStreamCreate(&stream_a);
    device_ping_kernel<<<1, 1, 0, stream_a>>>(flag, &time, &ret_ping);

    uint ret_pong;
    cudaSetDevice(device_b);
    cudaStreamCreate(&stream_b);
    device_pong_kernel<<<1, 1, 0, stream_b>>>(flag, &ret_pong);

    cudaDeviceSynchronize();
    cudaStreamDestroy(stream_b);
    cudaSetDevice(device_a);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream_a);
    cudaSetDevice(0);

    if (device_a == device_b) {
        printf("%u %u\n", ret_ping, ret_pong);
    }
    
    return (double) time / (double) get_gpu_clock_khz();
}

void run_ping_pong_benchmarks(size_t n_iter) {
    double times[n_iter];
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = host_ping_host_pong_benchmark(1, 2);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/host_host");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = host_ping_host_pong_benchmark(1, 72);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/host_remote_host");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = host_ping_host_pong_benchmark(72, 144);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/host_remote_host_far");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = host_ping_device_pong_benchmark(1, 0);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/host_device");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = device_ping_host_pong_benchmark(1, 0);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/device_host");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = host_ping_device_pong_benchmark(1, 1);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/host_remote_device");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = host_ping_device_pong_benchmark(72, 2);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/host_remote_device_far");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = device_ping_device_pong_benchmark(0, 0);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/device_device");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = device_ping_device_pong_benchmark(0, 1);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/device_remote_device");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = device_ping_device_pong_benchmark(1, 2);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/device_remote_device_far");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = device_ping_host_pong_benchmark(1, 1);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/device_remote_host");
    for (size_t iter = 0; iter < n_iter; ++iter) {
        times[iter] = device_ping_host_pong_benchmark(72, 2);
    }
    millisecond_times_to_latency_ns_file(times, n_iter, 10000, "results/pingpong/device_remote_host_far");
}