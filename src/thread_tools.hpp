#pragma once

#include "base_functions.hpp"
#include <iostream>
#include "base_kernels.cuh"

#define NUM_THREADS 15

constexpr size_t HOST_ID = (size_t) 0;
constexpr size_t DEVICE_ID = (size_t) -1;

class SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) {
            asm volatile("nop;nop;nop;nop;nop;nop;nop;nop; nop;nop;nop;nop;nop;nop;nop;nop;" :::);
        }
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
    TERMINATE
};


struct thread_data {
    std::unique_ptr<std::thread> t{nullptr};
    size_t t_id;
    SpinLock tx_mutex, rx_mutex;
    ThreadCommand command;
    uint8_t *buffer;
    size_t size;
} __attribute__((aligned(64)));

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
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto i = tid * 32; i < size; i += blockDim.x * gridDim.x * 32) {
        a[i] = 0;
    }
}

__global__ void device_read_preparation_kernel(uint8_t *a, size_t size) {
    __shared__ uint8_t dummy[2];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto i = tid * 32; i < size; i += blockDim.x * gridDim.x * 32) {
        dummy[i%2] = a[i];
    }

    if (tid < 0) {
        printf("%u", dummy[0]);
    }
}

void dispatch_memory_preparation(size_t t_id, ThreadCommand command, uint8_t *buffer, size_t size) {
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

void invalidate_all(uint8_t *buffer, size_t size) {
    prepare_memory(WRITE, buffer, size);        // put in OWNED state
    prepare_memory(READ, cache_filler, sizeof(cache_filler));   // invalidate locally
}

void thread_function(thread_data *t_info) {
    for (;;) {
        t_info->tx_mutex.lock(); // wait until main sends command
        switch (t_info->command) {
            case TERMINATE:
                return;
            default:
                prepare_memory(t_info->command, t_info->buffer, t_info->size);
        }
        t_info->rx_mutex.unlock(); // signal completion
    }
}

void init_thread_array() {
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        thread_data *cur = &thread_array[i];
        cur->t_id = i + 1;
        cur->tx_mutex.lock();
        cur->t = std::make_unique<std::thread>(thread_function, cur);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET((i + 1), &cpuset); 

        pthread_setaffinity_np(cur->t->native_handle(), sizeof(cpu_set_t), &cpuset);
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