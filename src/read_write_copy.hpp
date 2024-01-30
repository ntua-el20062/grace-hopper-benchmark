#pragma once

#include "data.hpp"
#include "base_kernels.cuh"
#include "measurement.hpp"
#include "thread_tools.hpp"
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void device_read_kernel_detailed(double *data, size_t n_elems, size_t n_iterations, clock_t *times) {
    double dummy[8];
    __shared__ clock_t times_vector[1024];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t iter = 0; iter < n_iterations; ++iter) {
        if (tid == 0) {
            times_vector[0] = 0;
        }
        __syncthreads();
        KERNEL_SYNC(times_vector);
        #pragma unroll 8
        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            dummy[i%8] = data[i];
        }
        KERNEL_MEASURE(times_vector);
        __syncthreads();
        if (threadIdx.x == 0) {
            clock_t elapsed = times_vector[0];
            for (size_t i = 1; i < blockDim.x; ++i) {
                elapsed = max(elapsed, times_vector[i]);
            }
            times[iter * gridDim.x + blockIdx.x] = elapsed;
        }
        // dummy print of element
        if (tid == -1) {
            printf("%lf ", dummy[0]);
        }
    }
}

__global__ void device_read_kernel(double *data, size_t n_elems) {
    double dummy;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
        dummy += data[i];
    }
    if (tid == -1) {
        printf("%lf ", dummy);
    }
}

__global__ void device_write_kernel(double *a, size_t n_elems) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
        a[i] = 0;
    }
}

__global__ void device_copy_kernel(double *a, double *b, size_t n_elems) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
        b[i] = a[i];
    }
}

__global__ void device_read_kernel_sync(double *data, size_t n_elems, size_t n_iter, clock_t *time, clock_t *sync) {
    double dummy;
    __shared__ clock_t clocks[1024];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    auto grid = cg::this_grid();

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        if (tid == 0) *sync = clock() + (19800000/7)*7;
        grid.sync();
        clock_t start = *sync;
        //if (threadIdx.x == 0) printf("%ld ", start);
        while (clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            dummy += data[i];
        }

        clock_t end = clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        grid.sync();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < 1024; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter * blockDim.x + blockIdx.x] = t;
        }
        // ----------------------
    }
    if (tid == -1) {
        printf("%lf ", dummy);
    }
}

__global__ void device_write_kernel_sync(double *a, size_t n_elems, size_t n_iter, clock_t *time, clock_t *sync) {
    __shared__ clock_t clocks[1024];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto grid = cg::this_grid();

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        if (tid == 0) *sync = clock() + (19800000/7)*7;
        grid.sync();
        clock_t start = *sync;
        while (clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            a[i] = 0;
        }
        
        clock_t end = clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        grid.sync();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < 1024; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter * blockDim.x + blockIdx.x] = t;
        }
        // ----------------------
    }
}

__global__ void device_copy_kernel_sync(double *a, double *b, size_t n_elems, size_t n_iter, clock_t *time, clock_t *sync) {
    __shared__ clock_t clocks[1024];
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto grid = cg::this_grid();


    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        if (tid == 0) *sync = clock() + (19800000/7)*7;
        grid.sync();
        clock_t start = *sync;
        while (clock() < start) {}
        // ------------------------

        for (auto i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            b[i] = a[i];
        }

        clock_t end = clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        grid.sync();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < 1024; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter * blockDim.x + blockIdx.x] = t;
        }
        // ----------------------
    }
}

__global__ void device_read_kernel_block(double *data, size_t n_elems, size_t n_iter, clock_t *time) {
    double dummy;
    __shared__ clock_t clocks[1025];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        clocks[1024] = clock() + (19800000/7)*7;
        __syncthreads();
        clock_t start = clocks[1024];
        while (clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            dummy += data[i];
        }

        clock_t end = clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        __syncthreads();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < 1024; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter] = t;
        }
        // ----------------------
    }
    if (tid == -1) {
        printf("%lf ", dummy);
    }
}

__global__ void device_write_kernel_block(double *a, size_t n_elems, size_t n_iter, clock_t *time) {
    __shared__ clock_t clocks[1025];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        clocks[1024] = clock() + (19800000/7)*7;
        __syncthreads();
        clock_t start = clocks[1024];
        while (clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            a[i] = 0;
        }
        
        clock_t end = clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        __syncthreads();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < 1024; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter] = t;
        }
        // ----------------------
    }
}

__global__ void device_copy_kernel_block(double *a, double *b, size_t n_elems, size_t n_iter, clock_t *time) {
    __shared__ clock_t clocks[1025];
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;


    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        clocks[1024] = clock() + (19800000/7)*7;
        __syncthreads();
        clock_t start = clocks[1024];
        while (clock() < start) {}
        // ------------------------

        for (auto i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            b[i] = a[i];
        }

        clock_t end = clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        __syncthreads();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < 1024; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter] = t;
        }
        // ----------------------
    }
}

__global__ void device_read_kernel_single(double *a, size_t n_elems, size_t n_iter, clock_t *time) {
    double dummy;
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        clock_t start = clock();
        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            dummy += a[i];
        }
        clock_t end = clock();
        time[iter] = end - start;
        if (tid == -1) {
            printf("%lf ", dummy);
        }
    }
}

__global__ void device_write_kernel_single(double *a, size_t n_elems, size_t n_iter, clock_t *time) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        clock_t start = clock();
        for (auto i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            a[i] = 0;
        }
        clock_t end = clock();
        time[iter] = end - start;
    }
}

__global__ void device_copy_kernel_single(double *a, double *b, size_t n_elems, size_t n_iter, clock_t *time) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        clock_t start = clock();
        for (auto i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            b[i] = a[i];
        }
        clock_t end = clock();
        time[iter] = end - start;
    }
}

template <typename ALLOC>
void write_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, size_t target, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    cudaDeviceReset();
    ALLOC factory(n_bytes);
    dispatch_command(target, command, factory.data, n_bytes);
    if (grid_size == 0) { // run with only one thread
        clock_t times[n_iter];
        device_write_kernel_single<<<1, 1>>>((double *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/write/device/single/" + name);
    } else if (grid_size == 1) { // run with only one block
        clock_t times[n_iter];
        device_write_kernel_block<<<1, 1024>>>((double *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/write/device/block/" + name);
    } else {
        clock_t big_times[n_iter * grid_size];
        clock_t small_times[n_iter];
        clock_t sync;
        void *args[] = {(void *) &factory.data, (void *) &n_elems, (void *) &n_iter, (void *) big_times, (void *) &sync};
        cudaLaunchCooperativeKernel((void *) device_write_kernel_sync, grid_size, 512, args, 0, 0);
        cudaDeviceSynchronize();
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/write/device/" + name);
        // for (size_t i = 0; i < n_iter; ++i) {
        //     times[i] = time_kernel_execution((void *) device_write_kernel, grid_size, 1024, args, 0, 0);
        // }
        // millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/write/device/" + name);
    }
}

template <typename ALLOC>
void read_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, size_t target, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    cudaDeviceReset();
    ALLOC factory(n_bytes);
    dispatch_command(target, command, factory.data, n_bytes);
    if (grid_size == 0) { // run with only one thread
        clock_t times[n_iter];
        device_read_kernel_single<<<1, 1>>>((double *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/read/device/single/" + name);
    } else if (grid_size == 1) { // run with only one block
        clock_t times[n_iter];
        device_read_kernel_block<<<1, 1024>>>((double *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/read/device/block/" + name);
    } else {
        clock_t big_times[n_iter * grid_size];
        clock_t small_times[n_iter];
        clock_t sync;
        void *args[] = {(void *) &factory.data, (void *) &n_elems, (void *) &n_iter, (void *) big_times, (void *) &sync};
        cudaLaunchCooperativeKernel((void *) device_read_kernel_sync, grid_size, 512, args, 0, 0);
        cudaDeviceSynchronize();
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/read/device/" + name);
        // for (size_t i = 0; i < n_iter; ++i) {
        //     times[i] = time_kernel_execution((void *) device_read_kernel, grid_size, 1024, args, 0, 0);
        // }
        // millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/read/device/" + name);
    }
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void copy_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    double times[n_iter];
    size_t per_array_bytes = n_bytes / 2;
    size_t n_elems = per_array_bytes / sizeof(double);

    cudaDeviceReset();
    SRC_ALLOC src(per_array_bytes);
    DST_ALLOC dst(per_array_bytes);
    dispatch_command(src_target, src_mode, src.data, per_array_bytes);
    dispatch_command(dst_target, dst_mode, dst.data, per_array_bytes);

    if (grid_size == 0) { // run with only one thread
        clock_t times[n_iter];
        device_copy_kernel_single<<<1, 1>>>((double *) src.data, (double *) dst.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/copy/device/single/" + name);
    } else if (grid_size == 1) { // run with only one block
        clock_t times[n_iter];
        device_copy_kernel_block<<<1, 1024>>>((double *) src.data, (double *) dst.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/copy/device/block/" + name);
    } else {
        clock_t big_times[n_iter * grid_size];
        clock_t small_times[n_iter];
        clock_t sync;
        void *args[] = {(void *) &src.data, (void *) &dst.data, (void *) &n_elems, (void *) &n_iter, (void *) big_times, (void *) &sync};
        cudaLaunchCooperativeKernel((void *) device_copy_kernel_sync, grid_size, 512, args, 0, 0);
        cudaDeviceSynchronize();
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/copy/device/" + name);
        // for (size_t i = 0; i < n_iter; ++i) {
        //     times[i] = time_kernel_execution((void *) device_copy_kernel, grid_size, 1024, args, 0, 0);
        // }
        // millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/copy/device/" + name);
    }
}

void run_write_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
    // write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "mmap_untouched");
    // write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, READ, "mmap_host_read");
    // write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, "mmap_host_written");
    // write_test_device_template<ManagedMemoryDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "managedmemory_untouched");
    // write_test_device_template<ManagedMemoryDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, "managedmemory_host_written");
    // write_test_device_template<CudaMallocHostDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "cudamallochost_untouched");
    // write_test_device_template<CudaMallocHostDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, "cudamallochost_host_written");
    // write_test_device_template<CudaMallocHostDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, READ, "cudamallochost_host_read");
    // write_test_device_template<CudaMallocDataFactory<>>(n_iter, n_bytes, grid_size, HOST_ID, NONE, "cudamalloc_untouched");

    write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, base + "ddr/" + end);
    write_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, DEVICE_ID, WRITE, base + "hbm/" + end);
    write_test_device_template<MmioDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, base + "mmio/" + end);
}

void run_read_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
    read_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, base + "ddr/" + end);
    read_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, DEVICE_ID, WRITE, base + "hbm/" + end);
    read_test_device_template<MmioDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, base + "mmio/" + end);
}

void run_copy_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, HOST_ID, WRITE, base + "ddr_ddr/" + end);
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, DEVICE_ID, WRITE, base + "ddr_hbm/" + end);
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, DEVICE_ID, WRITE, DEVICE_ID, WRITE, base + "hbm_hbm/" + end);
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, DEVICE_ID, WRITE, HOST_ID, WRITE, base + "hbm_ddr/" + end);
}


// ---------------- HOST FUNCTIONS --------------------------------
// __attribute__((always_inline)) inline double host_write_function(double *a, size_t n_elems, size_t n_threads) {
//     MEASURE_CPU_LOOP_AND_RETURN(
//         for (size_t i = 0; i < n_elems; ++i) {
//             a[i] = 0;
//         }
//     )
// }

// __attribute__((always_inline)) inline double host_read_function(double *a, size_t n_elems, size_t n_threads) {
//     MEASURE_CPU_LOOP_AND_RETURN(
//         for (size_t i = 0; i < n_elems; ++i) {
//             asm volatile("ldr x0, [%0]" :: "r" (&a[i]) : "x0");
//         }
//     )
// }

// __attribute__((always_inline)) inline double host_copy_function(double *a, double *b, size_t n_elems, size_t n_threads) {
//     MEASURE_CPU_LOOP_AND_RETURN(
//         for (size_t i = 0; i < n_elems; ++i) {
//             b[i] = a[i];
//         }
//     )
// }

// ---------------- HOST TEMPLATES --------------------------------
template <typename ALLOC>
void write_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t target, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_command(target, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = run_test(WRITE_TEST, n_threads, factory.data, nullptr, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/write/host/" + name);
}

template <typename ALLOC>
void read_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t device, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_command(device, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = run_test(WRITE_TEST, n_threads, factory.data, nullptr, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/read/host/" + name);
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void copy_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    double times[n_iter];
    size_t per_array_bytes = n_bytes / 2;
    size_t n_elems = per_array_bytes / sizeof(double);

    SRC_ALLOC src(per_array_bytes);
    DST_ALLOC dst(per_array_bytes);
    dispatch_command(src_target, src_mode, src.data, per_array_bytes);
    dispatch_command(dst_target, dst_mode, dst.data, per_array_bytes);

    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = run_test(WRITE_TEST, n_threads, src.data, dst.data, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/copy/host/" + name);
}

// ---------------- HOST TESTS --------------------------------
void run_write_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    // write_test_host_template<MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, NONE, "mmap_untouched");
    // write_test_host_template<MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, WRITE, "mmap_device_written");
    // write_test_host_template<MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, READ, "mmap_device_read");
    // write_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, DEVICE_ID, NONE, "managedmemory_untouched");
    // write_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, DEVICE_ID, WRITE, "managedmemory_device_written");
    // write_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, DEVICE_ID, READ, "managedmemory_device_read");
    write_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, base + "ddr/" + end);
    write_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, base + "hbm/" + end);
    write_test_host_template<MmioDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, NONE, base + "mmio/" + end);
}

void run_read_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    // read_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, NONE, "mmap_untouched");
    // read_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, "mmap_device_written");
    // read_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, READ, "mmap_device_read");
    // read_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, "mmap_host_written");
    // read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, NONE, "managedmemory_untouched");
    // read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, "managedmemory_device_written");
    // read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, "managedmemory_host_written");
    // read_test_host_template<ManagedMemoryDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, READ, "managedmemory_device_read");
    read_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, base + "ddr/" + end);
    read_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, base + "hbm/" + end);
    read_test_host_template<MmioDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, NONE, base + "mmio/" + end);
}

void run_copy_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, HOST_ID, WRITE, base + "ddr_ddr/" + end);
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, DEVICE_ID, WRITE, base + "ddr_hbm/" + end);
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, DEVICE_ID, WRITE, base + "hbm_hbm/" + end);
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, HOST_ID, WRITE, base + "hbm_ddr/" + end);
}