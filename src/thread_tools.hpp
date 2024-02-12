#pragma once

#include "base_functions.hpp"
#include <iostream>
#include "base_kernels.cuh"
#include "measurement.hpp"

#define NUM_THREADS 64

constexpr size_t HOST_ID = (size_t) 0;
constexpr size_t DEVICE_ID = (size_t) -1;

class SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) {}
    }
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

enum ThreadCommand {
    WRITE,
    READ,
    INVALIDATE,
    NONE,
    READ_TEST,
    WRITE_TEST,
    COPY_TEST,
    MEMSET_TEST,
    MEMCPY_TEST,
    TERMINATE
};


struct thread_data {
    std::unique_ptr<std::thread> t{nullptr};
    size_t t_id;
    SpinLock tx_mutex, rx_mutex;
    ThreadCommand command;
    uint8_t *buffer;
    uint8_t *second_buffer;
    size_t size;
    uint64_t start_time;
    uint64_t end_time;
} __attribute__((aligned(128)));

thread_data thread_array[NUM_THREADS];
uint8_t cache_filler[(int) ((double)CPU_L3_CACHE * 2.1f)];

__attribute__((noinline)) __host__ __device__ void dumb_copy(volatile uint8_t *source, volatile uint8_t *target, size_t size) {
    for (size_t i = 0; i < size; i += 64) {
        target[i] = source[i];
    }
}

void prepare_memory(ThreadCommand command, uint8_t *buffer, size_t size) {
    switch (command) {
        case WRITE: {
            //dumb_copy(buffer, buffer, size);
            for (size_t i = 0; i < size; i += 64) {
                asm volatile("ldr x0, [%0];"
                             "str x0, [%0]":: "r" (&buffer[i]) : "x0" );
            }
            break;
        }
        case READ: {
            for (size_t i = 0; i < size; i += 64) {
                asm volatile("ldr x0, [%0]" :: "r" (&buffer[i]) : "x0" );
            }
            break;
        }
        case INVALIDATE: {
            for (size_t i = 0; i < size; i += 64) {
                asm volatile("dc cvac, %0" : : "r"(&buffer[i]) : "memory");
            }
            for (size_t i = 0; i < sizeof(cache_filler); i += 64) {
                asm volatile("ldr x0, [%0]" :: "r" (&cache_filler[i]) : "x0", "memory");
            }
            for (size_t i = 0; i < sizeof(cache_filler); i += 64) {
                asm volatile("dc cvac, %0" : : "r"(&cache_filler[i]) : "memory");
            }
            __builtin___clear_cache(buffer, buffer + size);
            break;
        }
    }
    //asm volatile("dmb sy" ::: "memory");
}

__global__ void device_write_preparation_kernel(uint8_t *a, size_t size) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = tid * 32; i < size; i += blockDim.x * gridDim.x * 32) {
        a[i] = 0;
    }
}

__global__ void device_read_preparation_kernel(uint8_t *a, size_t size) {
    uint8_t dummy[2];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = tid * 32; i < size; i += blockDim.x * gridDim.x * 32) {
        dummy[i%2] = a[i];
    }

    if (tid == (size_t)-1) {
        printf("%u", dummy[0]);
    }
}

void dispatch_command(size_t t_id, ThreadCommand command, uint8_t *buffer, size_t size) {
    if (t_id == 0) {
        prepare_memory(command, buffer, size);
    } else if (t_id == DEVICE_ID) {
        switch (command) {
            case WRITE: {
                device_write_preparation_kernel<<<264, 1024>>>(buffer, size);
                cudaDeviceSynchronize();
                break;
            }
            case READ: {
                device_read_preparation_kernel<<<264, 1024>>>(buffer, size);
                cudaDeviceSynchronize();
                break;
            }
        }
    } else {
        thread_data *cur = &thread_array[t_id - 1];
        cur->command = command;
        cur->buffer = buffer;
        cur->size = size;
        cur->rx_mutex.lock();
        cur->tx_mutex.unlock(); // send command
        cur->rx_mutex.lock(); // wait until thread is done
        cur->rx_mutex.unlock();
    }
}

uint64_t run_test(ThreadCommand test, size_t n_threads, uint8_t *buffer, uint8_t *second_buffer, size_t n_elems, uint64_t *start_times, uint64_t *end_times) {
    assert(n_threads <= NUM_THREADS);
    size_t n_cachelines = n_elems / 8; // 8 doubles per cacheline
    size_t per_thread_n_cachelines = n_cachelines / n_threads;
    size_t remainder = n_cachelines % n_threads;
    uint64_t nominal_start_time = get_cpu_clock() + ((1000000/32)*32); // cur + 1ms (multiple of clock resolution)
    for (size_t i = 0; i < n_threads; ++i) {
        thread_data *cur = &thread_array[i];
        cur->command = test;
        cur->buffer = buffer;
        cur->second_buffer = second_buffer;
        cur->size = (per_thread_n_cachelines + (i < remainder ? 1 : 0)) * 8;
        cur->start_time = nominal_start_time;
        buffer += cur->size * sizeof(double);
        second_buffer += cur->size * sizeof(double);
    }

    for (size_t i = 0; i < n_threads; ++i) {
        thread_data *cur = &thread_array[i];
        cur->rx_mutex.lock();
        cur->tx_mutex.unlock(); // send command
    }

    for (size_t i = 0; i < n_threads; ++i) {
        thread_data *cur = &thread_array[i];
        cur->rx_mutex.lock(); // wait until thread is done
        cur->rx_mutex.unlock();
        start_times[i] = cur->start_time;
        end_times[i] = cur->end_time;
    }

    return nominal_start_time;
}

double time_test(ThreadCommand test, size_t n_threads, uint8_t *buffer, uint8_t *second_buffer, size_t n_elems) {
    uint64_t start_times[n_threads];
    uint64_t end_times[n_threads];

    uint64_t nominal_start_time = run_test(test, n_threads, buffer, second_buffer, n_elems, start_times, end_times);

    uint64_t max_end = 0;
    for (size_t i = 0; i < n_threads; ++i) {
        //max_end = std::max(max_end, end_times[i]);
        max_end += end_times[i];
    }
    max_end /= n_threads;

    return get_elapsed_milliseconds_clock(nominal_start_time, max_end);
}

void invalidate_all(uint8_t *buffer, size_t size) {
    prepare_memory(WRITE, buffer, size);        // put in OWNED state
    prepare_memory(READ, cache_filler, sizeof(cache_filler));   // invalidate locally
}

__attribute__((always_inline)) inline uint64_t thread_write_function(uint64_t start_time, double *a, size_t n_elems) {
    while (get_cpu_clock() < start_time) {}
    for (size_t outer = 0; outer < 1024; ++outer) {
        for (size_t i = 0; i < n_elems; i += 8) {
            asm volatile("str x0, [%0];"
                        "str x0, [%0, #8];"
                        "str x0, [%0, #16];"
                        "str x0, [%0, #24];"

                        "str x0, [%0, #32];"
                        "str x0, [%0, #40];"
                        "str x0, [%0, #48];"
                        "str x0, [%0, #56];" :: "r" (&a[i]) :);
        }
    }
    return get_cpu_clock();
}

__attribute__((always_inline)) inline uint64_t thread_memset_function(uint64_t start_time, double *a, size_t n_elems) {
    while (get_cpu_clock() < start_time) {}
    memset(a, 0xff, n_elems * sizeof(double)); // 0xff since setting to 0x00 does weird stuff
    return get_cpu_clock();
}

__attribute__((always_inline)) inline uint64_t thread_read_function(uint64_t start_time, double *a, size_t n_elems) {
    while (get_cpu_clock() < start_time) {}
    for (size_t outer = 0; outer < 1024; ++outer) {
        for (size_t i = 0; i < n_elems; i += 8) {
            asm volatile("ldr x0, [%0];"
                        "ldr x0, [%0, #8];"
                        "ldr x0, [%0, #16];"
                        "ldr x0, [%0, #24];"

                        "ldr x0, [%0, #32];"
                        "ldr x0, [%0, #40];"
                        "ldr x0, [%0, #48];"
                        "ldr x0, [%0, #56];" :: "r" (&a[i]) : "x0");
        }
    }
    return get_cpu_clock();
}

__attribute__((always_inline)) inline uint64_t thread_copy_function(uint64_t start_time, double *a, double *b, size_t n_elems) {
    while (get_cpu_clock() < start_time) {}
    for (size_t outer = 0; outer < 1024; ++outer) {
        for (size_t i = 0; i < n_elems; i += 8) {
            b[i] = a[i];
            b[i + 1] = a[i + 1];
            b[i + 2] = a[i + 2];
            b[i + 3] = a[i + 3];

            b[i + 4] = a[i + 4];
            b[i + 5] = a[i + 5];
            b[i + 6] = a[i + 6];
            b[i + 7] = a[i + 7];
        }
    }
    return get_cpu_clock();
}

__attribute__((always_inline)) inline uint64_t thread_memcpy_function(uint64_t start_time, double *a, double *b, size_t n_elems) {
    while (get_cpu_clock() < start_time) {}
    memcpy(b, a, n_elems * sizeof(double));
    return get_cpu_clock();
}

void thread_function(thread_data *t_info) {
    for (;;) {
        t_info->tx_mutex.lock(); // wait until main sends command
        switch (t_info->command) {
            case TERMINATE:
                return;
            case READ:
            case WRITE:
            case INVALIDATE:
                prepare_memory(t_info->command, t_info->buffer, t_info->size);
                break;
            case READ_TEST:
                t_info->end_time = thread_read_function(t_info->start_time, (double *)t_info->buffer, t_info->size);
                break;
            case WRITE_TEST:
                t_info->end_time = thread_write_function(t_info->start_time, (double *)t_info->buffer, t_info->size);
                break;
            case COPY_TEST:
                t_info->end_time = thread_copy_function(t_info->start_time, (double *)t_info->buffer, (double *) t_info->second_buffer, t_info->size);
                break;
            case MEMSET_TEST:
                t_info->end_time = thread_memset_function(t_info->start_time, (double *)t_info->buffer, t_info->size);
                break;
            case MEMCPY_TEST:
                t_info->end_time = thread_memcpy_function(t_info->start_time, (double *)t_info->buffer, (double *)t_info->second_buffer, t_info->size);
                break;
        }
        t_info->rx_mutex.unlock(); // signal completion
    }
}

void init_thread_array() {
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        thread_data *cur = &thread_array[i];
        cur->t_id = i + 1;
        cur->start_time = 0;
        cur->tx_mutex.lock();
        cur->t = std::make_unique<std::thread>(thread_function, cur);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET((i + 1), &cpuset); 

        pthread_setaffinity_np(cur->t->native_handle(), sizeof(cpu_set_t), &cpuset);
        struct sched_param param;
        param.sched_priority = 1;
        pthread_setschedparam(cur->t->native_handle(), SCHED_FIFO, &param);
    }
}

void terminate_threads() {
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        thread_data *cur = &thread_array[i];

        cur->command = TERMINATE;
        cur->tx_mutex.unlock();
        cur->t->join();
    }
}