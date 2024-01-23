#pragma once

#include "data.hpp"
#include "base_kernels.cuh"
#include "measurement.hpp"
#include "thread_tools.hpp"

template <typename T>
__attribute__((always_inline)) inline void host_write_function(T *a, size_t n_elems) {
    #pragma omp parallel for
    for (size_t i = 0; i < n_elems; ++i) {
        a[i] = 0;
    }
}

template <typename T>
__attribute__((always_inline)) inline double host_read_function(T *a, size_t n_elems) {
#ifdef _OPENMP
    volatile uint64_t times[omp_get_max_threads()];
    times[0] = 0;
    #pragma omp parallel
    {
        OMP_SYNC(times);
        #pragma omp for
        for (size_t i = 0; i < n_elems; ++i) {
            asm volatile("ldr x0, [%0]" :: "r" (&a[i]) : "x0");
        }
        OMP_MEASURE(times);
    }

    volatile uint64_t out_time = times[0];
    for (size_t i = 0; i < omp_get_max_threads(); ++i) {
        out_time = std::max(out_time, times[i]);
    }
#else
    auto start = get_cpu_clock();
    for (size_t i = 0; i < n_elems; ++i) {
        asm volatile("ldr x0, [%0]" :: "r" (&a[i]) : "x0");
    }
    auto out_time = get_cpu_clock() - start;
#endif

    return clock_to_milliseconds(out_time);
}

template <typename ALLOC>
void write_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, size_t target, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    cudaDeviceReset();
    ALLOC factory(n_bytes);
    dispatch_memory_preparation(target, command, factory.data, n_bytes);
    void *args[] = {(void *) &factory.data, (void *) &n_elems};
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_kernel_execution((void *) device_write_kernel<double>, grid_size, 1024, args, 0, 0);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/write/device/" + name + "/" + std::to_string(n_bytes));
}

void run_write_tests_device(size_t n_iter, size_t n_bytes, int grid_size) {
    write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "mmap_untouched");
    write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, READ, "mmap_host_read");
    write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, "mmap_host_written");
    write_test_device_template<ManagedMemoryDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "managedmemory_untouched");
    write_test_device_template<ManagedMemoryDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, "managedmemory_host_written");
    write_test_device_template<CudaMallocHostDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "cudamallochost_untouched");
    write_test_device_template<CudaMallocHostDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, "cudamallochost_host_written");
    write_test_device_template<CudaMallocHostDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, READ, "cudamallochost_host_read");
    write_test_device_template<CudaMallocDataFactory<>>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "cudamalloc_untouched");
}

template <typename ALLOC>
void write_test_host_template(size_t n_iter, size_t n_bytes, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_memory_preparation(DEVICE_ID, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_function_execution(host_write_function<double>, (double *) factory.data, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/write/host/" + name + "/" + std::to_string(n_bytes));
}

template <typename ALLOC>
void read_test_host_template(size_t n_iter, size_t n_bytes, size_t device, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_memory_preparation(device, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = host_read_function<double>((double *) factory.data, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/read/host/" + name + "/" + std::to_string(n_bytes));
}

void run_write_tests_host(size_t n_iter, size_t n_bytes) {
    write_test_host_template<MmapDataFactory>(n_iter, n_bytes, NONE, "mmap_untouched");
    write_test_host_template<MmapDataFactory>(n_iter, n_bytes, WRITE, "mmap_device_written");
    write_test_host_template<MmapDataFactory>(n_iter, n_bytes, READ, "mmap_device_read");
    write_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, NONE, "managedmemory_untouched");
    write_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, WRITE, "managedmemory_device_written");
    write_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, READ, "managedmemory_device_read");
}

void run_read_tests_host(size_t n_iter, size_t n_bytes) {
    read_test_host_template<MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, NONE, "mmap_untouched");
    read_test_host_template<MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, WRITE, "mmap_device_written");
    read_test_host_template<MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, READ, "mmap_device_read");
    read_test_host_template<MmapDataFactory>(n_iter, n_bytes, HOST_ID, WRITE, "mmap_host_written");
    read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, DEVICE_ID, NONE, "managedmemory_untouched");
    read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, DEVICE_ID, WRITE, "managedmemory_device_written");
    read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, HOST_ID, WRITE, "managedmemory_host_written");
    read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, DEVICE_ID, READ, "managedmemory_device_read");
}