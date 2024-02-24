#pragma once

#include "thread_tools.hpp"
#include "data.hpp"

__attribute__((always_inline)) inline double latency_function(uint8_t *in, size_t n_elem) {    
    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < n_elem/8; ++i) {
        asm volatile("ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"

                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];"
                     "ldr %0, [%1];" : "=r" (in) : "r" ((uint8_t **) in) :);
    }
    uint64_t end = get_cpu_clock();

    return get_elapsed_milliseconds_clock(start, end);
}

__attribute__((always_inline)) inline double latency_write_function(uint8_t *in, size_t n_elem) {    
    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < n_elem/8; ++i) {
        asm volatile("ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"

                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];"
                     "str x0, [%1, #8];"
                     "ldr %0, [%1];" 
                     "str x0, [%1, #8];" : "=r" (in) : "r" ((uint8_t **) in) :);
    }
    uint64_t end = get_cpu_clock();

    return get_elapsed_milliseconds_clock(start, end);
}

__global__ void latency_kernel(uint8_t *in, size_t n_elem, size_t n_iter, double *time) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        uint8_t *itr = in;
        clock_t start = clock();
        for (size_t i = 0; i < n_elem/8; ++i) {
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);

            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
            itr = *((uint8_t **) itr);
        }
        clock_t end = clock();
        time[iter] = end - start;

        if (tid > 1) { // dummy work
            printf("%u ", *itr);
        }

    }
}

__global__ void latency_write_kernel(uint8_t *in, size_t n_elem, size_t n_iter, double *time) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        uint8_t *itr = in;
        clock_t start = clock();
        for (size_t i = 0; i < n_elem/8; ++i) {
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;

            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
            itr = *((uint8_t **) itr);
            *((size_t *) itr + 1) = n_iter;
        }
        clock_t end = clock();
        time[iter] = end - start;

        if (tid > 1) { // dummy work
            printf("%u ", *itr);
        }

    }
}

template <bool IS_PAGE, bool IS_WRITE, typename ALLOC>
void latency_test_host_template(size_t n_iter, size_t n_bytes, std::string name) {
    double times[n_iter];

    ALLOC factory(n_bytes);
    size_t n_elem;
    if constexpr (IS_PAGE) {
        initialize_memory_page_chase(factory.data, n_bytes);
        n_elem = n_bytes / PAGE_SIZE;
    } else {
        initialize_memory_pointer_chase(factory.data, n_bytes);
        n_elem = n_bytes / CACHELINE_SIZE;
    }

    for (size_t i = 0; i < n_iter; ++i) {
        if constexpr (IS_WRITE) {
            times[i] = latency_write_function(factory.data, n_elem);
        } else {
            times[i] = latency_function(factory.data, n_elem);
        }
    }

    millisecond_times_to_latency_ns_file(times, n_iter, n_elem, name);
}

template <bool IS_PAGE>
void run_latency_test_host(size_t n_iter, size_t n_bytes) {
    std::string base = "results/latency/";
    if (!IS_PAGE) {
        base += "fine/";
    }
    latency_test_host_template<IS_PAGE, false, HOST_MEM>(n_iter, n_bytes, base + "host/ddr");
    latency_test_host_template<IS_PAGE, false, DEVICE_MEM>(n_iter, n_bytes, base + "host/hbm");
    latency_test_host_template<IS_PAGE, false, REMOTE_HOST_MEM>(n_iter, n_bytes, base + "host/ddr_remote");
    latency_test_host_template<IS_PAGE, false, REMOTE_DEVICE_MEM>(n_iter, n_bytes, base + "host/hbm_remote");
}

template <bool IS_PAGE, bool IS_WRITE, typename ALLOC>
void latency_test_device_template(size_t n_iter, size_t n_bytes, std::string name) {
    double gpu_clock = get_gpu_clock_khz();
    double times[n_iter];

    ALLOC factory(n_bytes);
    size_t n_elem;
    if constexpr (IS_PAGE) {
        initialize_memory_page_chase(factory.data, n_bytes);
        n_elem = n_bytes / PAGE_SIZE;
    } else {
        initialize_memory_pointer_chase(factory.data, n_bytes);
        n_elem = n_bytes / CACHELINE_SIZE;
    }

    if constexpr (IS_WRITE) {
        latency_write_kernel<<<1, 1>>>(factory.data, n_elem, n_iter, times);
    } else {
        latency_kernel<<<1, 1>>>(factory.data, n_elem, n_iter, times);
    }
    cudaDeviceSynchronize();

    for (size_t i = 0; i < n_iter; ++i) {
        times[i] /= gpu_clock;
    }

    millisecond_times_to_latency_ns_file(times, n_iter, n_elem, name);
}

template <bool IS_PAGE>
void run_latency_test_device(size_t n_iter, size_t n_bytes) {
    std::string base = "results/latency/";
    if (!IS_PAGE) {
        base += "fine/";
    }
    latency_test_device_template<IS_PAGE, false, HOST_MEM>(n_iter, n_bytes, base + "device/ddr");
    latency_test_device_template<IS_PAGE, false, DEVICE_MEM>(n_iter, n_bytes, base + "device/hbm");
    latency_test_device_template<IS_PAGE, false, REMOTE_HOST_MEM>(n_iter, n_bytes, base + "device/ddr_remote");
    latency_test_device_template<IS_PAGE, false, REMOTE_DEVICE_MEM>(n_iter, n_bytes, base + "device/hbm_remote");
}
