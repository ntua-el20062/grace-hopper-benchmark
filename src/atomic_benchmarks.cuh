#pragma once

#include <cuda/std/atomic>
#include "measurement.hpp"

#define TIMES10(a) a a a a a a a a a a
#define TIMES100(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a) TIMES10(a)
#define TIMES1000(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a) TIMES100(a)

#define FLAG_A 0
#define FLAG_B 1

__global__ void device_pong_kernel(cuda::std::atomic<int> &flag) {
    flag.store(FLAG_A, cuda::std::memory_order_relaxed);
    for (size_t i = 0; i < 1000; ++i) {
        while (flag.load(cuda::std::memory_order_relaxed) == FLAG_A);
        flag.store(FLAG_A, cuda::std::memory_order_relaxed);
    }
}

void host_pong_function(std::atomic<int> &flag) {
    flag.store(FLAG_A, std::memory_order_relaxed);
    for (size_t i = 0; i < 1000; ++i) {
        while (flag.load(std::memory_order_relaxed) == FLAG_A);
        flag.store(FLAG_A, std::memory_order_relaxed);
    }
}

float host_ping_device_pong_benchmark() {
    struct timeval start, end;

    cuda::std::atomic<int> flag{FLAG_B};

    device_pong_kernel<<<1, 1>>>(flag);

    // synchronize at the start
    while (flag.load(cuda::std::memory_order_relaxed) == FLAG_B);

    gettimeofday(&start, nullptr);
    for (size_t i = 0; i < 1000; ++i) {
        flag.store(FLAG_B, cuda::std::memory_order_relaxed);
        while (flag.load(cuda::std::memory_order_relaxed) == FLAG_B);
    }
    gettimeofday(&end, nullptr);

    return get_elapsed_milliseconds(start, end);
}

float host_ping_host_pong_benchmark() {
    struct timeval start, end;

    std::atomic<int> flag{FLAG_B};

    std::thread t(host_pong_function, std::ref(flag));
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);  // set affinity to CPU 1

    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);

    // synchronize at the start
    while (flag.load(std::memory_order_relaxed) == FLAG_B);

    gettimeofday(&start, nullptr);
    for (size_t i = 0; i < 1000; ++i) {
        flag.store(FLAG_B, std::memory_order_relaxed);
        while (flag.load(std::memory_order_relaxed) == FLAG_B);
    }
    gettimeofday(&end, nullptr);

    t.join();

    return get_elapsed_milliseconds(start, end);
}