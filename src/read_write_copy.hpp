#pragma once

#include "data.hpp"
#include "base_kernels.cuh"
#include "measurement.hpp"
#include "thread_tools.hpp"
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void device_write_kernel_flat(double *a) {
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    a[tid] = 0;
}

__global__ void device_copy_kernel_flat(double *a, double *b) {
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    b[tid] = a[tid];
}

__global__ void device_read_kernel_detailed(double *data, size_t n_elems, size_t n_iterations, clock_t *times) {
    double dummy[8];
    __shared__ clock_t times_vector[1024];

    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

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
        if (tid == (size_t)-1) {
            printf("%lf ", dummy[0]);
        }
    }
}

__global__ void device_read_kernel(double *data, size_t n_elems) {
    double dummy;

    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    #pragma unroll 8
    for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
        dummy += data[i];
    }
    if (tid == (size_t)-1) {
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
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
        b[i] = a[i];
    }
}

__global__ void device_read_kernel_sync(double *data, size_t n_elems, size_t n_iter, clock_t *time, clock_t *sync) {
    uint64_t dummy;
    __shared__ clock_t clocks[1024];

    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    auto grid = cg::this_grid();

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        *sync = get_gpu_clock() + (1000000/32)*32;
        grid.sync();
        __syncthreads();
        clock_t start = *sync;
        while (get_gpu_clock() < start) {}
        // ------------------------

        #pragma unroll 8
        for (size_t outer = 0; outer < 1024; ++outer) {
            for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
                uint64_t v;
                asm volatile("ld.ca.u64 %0, [%1];" : "=l"(v) : "l"(&data[i]));
                dummy ^= v;
            }
        }

        clock_t end = get_gpu_clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        grid.sync();

        if (threadIdx.x == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < blockDim.x; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter * gridDim.x + blockIdx.x] = t;
        }
        // ----------------------
    }
    if (*sync == 0) {
        printf("%lu ", dummy);
    }
}

__global__ void device_write_kernel_sync(double *a, size_t n_elems, size_t n_iter, clock_t *time, clock_t *sync) {
    __shared__ clock_t clocks[1024];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto grid = cg::this_grid();

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        *sync = get_gpu_clock() + (1000000/32)*32;
        grid.sync();
        __syncthreads();
        clock_t start = *sync;
        while (get_gpu_clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            a[i] = 0;
        }
        
        clock_t end = get_gpu_clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        grid.sync();
        if (threadIdx.x == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < blockDim.x; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter * gridDim.x + blockIdx.x] = t;
        }
        // ----------------------
    }
}

__global__ void device_write_kernel_time_analysis(double *a, size_t n_elems, size_t n_iter, clock_t *start_times, clock_t *end_times, clock_t *sync) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto grid = cg::this_grid();

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        *sync = get_gpu_clock() + (1000000/32)*32;
        grid.sync();
        __syncthreads();
        clock_t start = *sync;
        clock_t local_start;
        do {
            local_start = get_gpu_clock();
        } while (local_start < start);
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            a[i] = 0;
        }
        
        clock_t end = get_gpu_clock();

        // STORE -----------------
        grid.sync();
        start_times[iter * gridDim.x * blockDim.x + tid] = local_start - start;
        end_times[iter * gridDim.x * blockDim.x + tid] = end - start;
        // ----------------------
    }
}

void run_clock_analysis_device() {
    size_t grid_size = 264;
    size_t block_size = 1024;
    size_t n_iter = 10;
    clock_t *start_times = (clock_t *) malloc(sizeof(clock_t) * n_iter * grid_size * block_size);
    clock_t *end_times = (clock_t *) malloc(sizeof(clock_t) * n_iter * grid_size * block_size);
    clock_t sync;
    clock_t *sync_ptr = &sync;
    size_t n_bytes = 1UL << 32;
    size_t n_elems = n_bytes / sizeof(double);
    MmapDataFactory factory(n_bytes);
    dispatch_command(DEVICE_ID, WRITE, factory.data, n_bytes);
    void *args[] = {(void *) &factory.data, (void *) &n_elems, (void *) &n_iter, (void *) &start_times, (void *) &end_times, (void *) &sync_ptr};
    cudaLaunchCooperativeKernel((void *) device_write_kernel_time_analysis, grid_size, block_size, args, 0, 0);
    cudaDeviceSynchronize();
    for (size_t itr = 0; itr < n_iter; itr++) {
        std::ofstream start_file("results/clock_analysis/device/start/" + std::to_string(itr));
        std::ofstream end_file("results/clock_analysis/device/end/" + std::to_string(itr));
        for (size_t i = 0; i < grid_size * block_size; i++) {
            start_file << start_times[itr * grid_size * block_size + i] << std::endl;
            end_file << end_times[itr * grid_size * block_size + i] << std::endl;
        }
    }

    free(start_times);
    free(end_times);
}

void run_clock_analysis_host() {
    size_t n_threads = 2;
    size_t n_iter = 10;
    uint64_t start_times[n_iter * n_threads];
    uint64_t end_times[n_iter * n_threads];
    size_t n_bytes = 1UL << 32;
    size_t n_elems = n_bytes / sizeof(double);
    MmapDataFactory factory(n_bytes);
    dispatch_command(HOST_ID, WRITE, factory.data, n_bytes);
    for (size_t itr = 0; itr < n_iter; itr++) {
        uint64_t nominal_time = run_test(WRITE_TEST, n_threads, factory.data, nullptr, n_elems, &start_times[itr * n_threads], &end_times[itr * n_threads]);
        for (size_t j = 0; j < n_threads; ++j) {
            start_times[itr * n_threads + j] -= nominal_time;
            end_times[itr * n_threads + j] -= nominal_time;
        }
    }
    for (size_t itr = 0; itr < n_iter; itr++) {
        std::ofstream start_file("results/clock_analysis/host/start/" + std::to_string(itr));
        std::ofstream end_file("results/clock_analysis/host/end/" + std::to_string(itr));
        for (size_t i = 0; i < n_threads; i++) {
            start_file << start_times[itr * n_threads + i] << std::endl;
            end_file << end_times[itr * n_threads + i] << std::endl;
        }
    }
}

__global__ void device_copy_kernel_sync(double *a, double *b, size_t n_elems, size_t n_iter, clock_t *time, clock_t *sync) {
    __shared__ clock_t clocks[1024];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto grid = cg::this_grid();


    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        *sync = get_gpu_clock() + (1000000/32)*32;
        grid.sync();
        __syncthreads();
        clock_t start = *sync;
        while (get_gpu_clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            b[i] = a[i];
        }

        clock_t end = get_gpu_clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        grid.sync();
        if (threadIdx.x == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < blockDim.x; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter * gridDim.x + blockIdx.x] = t;
        }
        // ----------------------
    }
}

__global__ void device_read_kernel_block(uint64_t *data, size_t n_elems, size_t n_iter, clock_t *time) {
    uint64_t dummy[8];
    __shared__ clock_t clocks[1025];

    assert(blockIdx.x == 0);

    size_t tid = threadIdx.x;

    size_t per_thread = n_elems / 1024;

    for (size_t i = tid; i < n_elems; i += 1024) {
        dummy[0] ^= data[i];
    }

    uint64_t *ptr = data + tid;
    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        if (tid == 0) clocks[1024] = clock() + (19800000/7)*7;
        __syncthreads();
        clock_t start = clocks[1024];
        while (clock() < start) {}
        // ------------------------

        for (size_t j = 0; j < 8192; ++j) {
            for (size_t i = 0; i < per_thread; i += 8) {
                dummy[0] ^= ptr[i];
                dummy[1] ^= ptr[i + 1];
                dummy[2] ^= ptr[i + 2];
                dummy[3] ^= ptr[i + 3];

                dummy[4] ^= ptr[i + 4];
                dummy[5] ^= ptr[i + 5];
                dummy[6] ^= ptr[i + 6];
                dummy[7] ^= ptr[i + 7];
            }
        }

        clock_t end = clock();
        clocks[threadIdx.x] = end;

        // STORE -----------------
        __syncthreads();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < blockDim.x; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter] = (t - start) / 8192;
        }
        // ----------------------
    }
    if (tid > clocks[tid]) {
        printf("%lu ", dummy[tid%8]);
    }
}

__global__ void device_write_kernel_block(double *a, size_t n_elems, size_t n_iter, clock_t *time) {
    __shared__ clock_t clocks[1025];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        if (tid == 0) clocks[1024] = clock() + (19800000/7)*7;
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
            for (size_t i = 1; i < blockDim.x; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter] = t;
        }
        // ----------------------
    }
}

__global__ void device_copy_kernel_block(double *a, double *b, size_t n_elems, size_t n_iter, clock_t *time) {
    __shared__ clock_t clocks[1025];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;


    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        if (tid == 0) clocks[1024] = clock() + (19800000/7)*7;
        __syncthreads();
        clock_t start = clocks[1024];
        while (clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
            b[i] = a[i];
        }

        clock_t end = clock();
        clocks[threadIdx.x] = end - start;

        // STORE -----------------
        __syncthreads();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < blockDim.x; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter] = t;
        }
        // ----------------------
    }
}

__global__ void device_read_kernel_single(uint64_t *a, size_t n_elems, size_t n_iter, volatile clock_t *time) {
    uint64_t dummy[8];
    

    for (size_t iter = 0; iter < n_iter; ++iter) {
        clock_t start = clock();
        for (size_t i = 0; i < n_elems; i += 8) {
            dummy[0] ^= a[i];
            dummy[1] ^= a[i+1];
            dummy[2] ^= a[i+2];
            dummy[3] ^= a[i+3];
            dummy[4] ^= a[i+4];
            dummy[5] ^= a[i+5];
            dummy[6] ^= a[i+6];
            dummy[7] ^= a[i+7];
        }
        clock_t end = clock();
        time[iter] = end - start;
        if (*time < 8) {
            printf("%lu ", dummy[*time]);
        }
    }
}

__global__ void device_write_kernel_single(double *a, size_t n_elems, size_t n_iter, clock_t *time) {
    for (size_t iter = 0; iter < n_iter; ++iter) {
        clock_t start = clock();
        for (size_t i = 0; i < n_elems; i += 8) {
            a[i] = i;
            a[i+1] = i;
            a[i+2] = i;
            a[i+3] = i;
            a[i+4] = i;
            a[i+5] = i;
            a[i+6] = i;
            a[i+7] = i;
        }
        clock_t end = clock();
        time[iter] = end - start;
    }
}

__global__ void device_copy_kernel_single(double *a, double *b, size_t n_elems, size_t n_iter, clock_t *time) {
    for (size_t iter = 0; iter < n_iter; ++iter) {
        clock_t start = clock();
        for (size_t i = 0; i < n_elems; i += 8) {
            b[i] = a[i];
            b[i+1] = a[i+1];
            b[i+2] = a[i+2];
            b[i+3] = a[i+3];

            b[i+4] = a[i+4];
            b[i+5] = a[i+5];
            b[i+6] = a[i+6];
            b[i+7] = a[i+7];
        }
        clock_t end = clock();
        time[iter] = end - start;
    }
}

template <typename ALLOC>
void write_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, size_t target, ThreadCommand command, std::string name) {
    //double times[n_iter];
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
        clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * grid_size);
        clock_t small_times[n_iter];
        clock_t sync;
        clock_t *sync_ptr = &sync;
        void *args[] = {(void *) &factory.data, (void *) &n_elems, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
        gpuErrchk(cudaLaunchCooperativeKernel((void *) device_write_kernel_sync, grid_size, 1024, args, 0, 0));
        gpuErrchk(cudaDeviceSynchronize());
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/write/device/" + name, 1000000000.);
        // for (size_t i = 0; i < n_iter; ++i) {
        //     times[i] = time_kernel_execution((void *) device_write_kernel, grid_size, 1024, args, 0, 0);
        // }
        // millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/write/device/" + name);
    }
}

template <typename ALLOC>
void read_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, size_t target, ThreadCommand command, std::string name) {
    //double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    cudaDeviceReset();
    ALLOC factory(n_bytes);
    dispatch_command(target, command, factory.data, n_bytes);
    // cudaMemPrefetchAsync(factory.data, n_bytes, 0, 0);
    if (grid_size == 0) { // run with only one thread
        clock_t times[n_iter];
        device_read_kernel_single<<<1, 1>>>((uint64_t *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/read/device/single/" + name);
    } else if (grid_size == 1) { // run with only one block
        clock_t times[n_iter];
        cudaMemset(factory.data, 0x00, n_bytes);
        device_read_kernel_block<<<1, 1024>>>((uint64_t *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/read/device/block/" + name);
    } else {
        clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * grid_size);
        clock_t small_times[n_iter];
        clock_t sync;
        clock_t *sync_ptr = &sync;
        void *args[] = {(void *) &factory.data, (void *) &n_elems, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
        cudaLaunchCooperativeKernel((void *) device_read_kernel_sync, grid_size, 1024, args, 0, 0);
        gpuErrchk(cudaDeviceSynchronize());
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t / 1024;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/read/device/" + name, 1000000000.);
        // for (size_t i = 0; i < n_iter; ++i) {
        //     times[i] = time_kernel_execution((void *) device_read_kernel, grid_size, 1024, args, 0, 0);
        // }
        // millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/read/device/" + name);
    }
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void copy_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    //double times[n_iter];
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
        times_to_file(times, n_iter, per_array_bytes, "results/copy/device/single/" + name);
    } else if (grid_size == 1) { // run with only one block
        clock_t times[n_iter];
        device_copy_kernel_block<<<1, 1024>>>((double *) src.data, (double *) dst.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, per_array_bytes, "results/copy/device/block/" + name);
    } else {
        clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * grid_size);
        clock_t small_times[n_iter];
        clock_t sync;
        clock_t *sync_ptr = &sync;
        void *args[] = {(void *) &src.data, (void *) &dst.data, (void *) &n_elems, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
        cudaLaunchCooperativeKernel((void *) device_copy_kernel_sync, grid_size, 1024, args, 0, 0);
        gpuErrchk(cudaDeviceSynchronize());
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, per_array_bytes, "results/copy/device/" + name, 1000000000.);
        // for (size_t i = 0; i < n_iter; ++i) {
        //     times[i] = time_kernel_execution((void *) device_copy_kernel, grid_size, 1024, args, 0, 0);
        // }
        // millisecond_times_to_gb_sec_file(times, n_iter, per_array_bytes, "results/copy/device/" + name);
    }
}



void write_throughput_test_device_template(size_t n_iter, size_t n_bytes, size_t device, ThreadCommand command, std::string name) {
    size_t n_elems = n_bytes / sizeof(double);
    int grid_size = n_elems / 1024;
    double times[n_iter];

    MmapDataFactory factory(n_bytes);
    dispatch_command(device, command, factory.data, n_bytes);
    void *args[] = {(void *) &factory.data};
    for (size_t itr = 0; itr < n_iter; ++itr) {
        times[itr] = time_kernel_execution((void *) device_write_kernel_flat, grid_size, 1024, args, 0, 0);
    }

    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/write/device/throughput/" + name);
}

void run_write_throughput_test_device(size_t n_iter, size_t n_bytes) {
    write_throughput_test_device_template(n_iter, n_bytes, HOST_ID, WRITE, "ddr/" + std::to_string(n_bytes));
    write_throughput_test_device_template(n_iter, n_bytes, DEVICE_ID, WRITE, "hbm/" + std::to_string(n_bytes));
}

void copy_throughput_test_device_template(size_t n_iter, size_t n_bytes, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    size_t n_elems = n_bytes / sizeof(double);
    int grid_size = n_elems / 1024;
    double times[n_iter];

    MmapDataFactory src(n_bytes);
    MmapDataFactory dst(n_bytes);
    dispatch_command(src_target, src_mode, src.data, n_bytes);
    dispatch_command(dst_target, dst_mode, dst.data, n_bytes);
    void *args[] = {(void *) &src.data, (void *) &dst.data};
    for (size_t itr = 0; itr < n_iter; ++itr) {
        times[itr] = time_kernel_execution((void *) device_copy_kernel_flat, grid_size, 1024, args, 0, 0);
    }

    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/copy/device/throughput/" + name);
}

void run_copy_throughput_test_device(size_t n_iter, size_t n_bytes) {
    copy_throughput_test_device_template(n_iter, n_bytes, HOST_ID, WRITE, HOST_ID, WRITE, "ddr_ddr/" + std::to_string(n_bytes));
    copy_throughput_test_device_template(n_iter, n_bytes, HOST_ID, WRITE, DEVICE_ID, WRITE, "ddr_hbm/" + std::to_string(n_bytes));
    copy_throughput_test_device_template(n_iter, n_bytes, DEVICE_ID, WRITE, HOST_ID, WRITE, "hbm_ddr/" + std::to_string(n_bytes));
    copy_throughput_test_device_template(n_iter, n_bytes, DEVICE_ID, WRITE, DEVICE_ID, WRITE, "hbm_hbm/" + std::to_string(n_bytes));
}

void run_write_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
    //int block_size;
    //gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, device_write_kernel_sync));
    //std::cout << "write kernel: " << grid_size << " " << block_size << std::endl;
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
    // write_test_device_template<MmioDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, base + "mmio/" + end);
}

void run_read_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
        //int block_size;
    //gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, device_read_kernel_sync));
    //std::cout << "read kernel: " << grid_size << " " << block_size << std::endl;
    read_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, base + "ddr/" + end);
    read_test_device_template<MmapDataFactory>(n_iter, n_bytes, grid_size, DEVICE_ID, WRITE, base + "hbm/" + end);
    // read_test_device_template<MmioDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, NONE, base + "mmio/" + end);
}

void run_copy_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
        //int block_size;
    //gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, device_copy_kernel_sync));
    //std::cout << "copy kernel: " << grid_size << " " << block_size << std::endl;
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, HOST_ID, WRITE, base + "ddr_ddr/" + end);
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, HOST_ID, WRITE, DEVICE_ID, WRITE, base + "ddr_hbm/" + end);
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, DEVICE_ID, WRITE, DEVICE_ID, WRITE, base + "hbm_hbm/" + end);
    copy_test_device_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, grid_size, DEVICE_ID, WRITE, HOST_ID, WRITE, base + "hbm_ddr/" + end);
}

// ---------------- HOST TEMPLATES --------------------------------
template <typename ALLOC>
void write_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t target, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_command(target, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(WRITE_TEST, n_threads, factory.data, nullptr, n_elems) / 1024;
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/write/host/" + name);
}

template <typename ALLOC>
void memset_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t target, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_command(target, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(MEMSET_TEST, n_threads, factory.data, nullptr, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/memset/host/" + name);
}

template <typename ALLOC>
void read_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t device, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_command(device, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(READ_TEST, n_threads, factory.data, nullptr, n_elems) / 1024;
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
        times[i] = time_test(COPY_TEST, n_threads, src.data, dst.data, n_elems) / 1024;
    }
    millisecond_times_to_gb_sec_file(times, n_iter, per_array_bytes, "results/copy/host/" + name);
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void memcpy_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    double times[n_iter];
    size_t per_array_bytes = n_bytes / 2;
    size_t n_elems = per_array_bytes / sizeof(double);

    SRC_ALLOC src(per_array_bytes);
    DST_ALLOC dst(per_array_bytes);
    dispatch_command(src_target, src_mode, src.data, per_array_bytes);
    dispatch_command(dst_target, dst_mode, dst.data, per_array_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(MEMCPY_TEST, n_threads, src.data, dst.data, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, per_array_bytes, "results/memcpy/host/" + name);
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
    // write_test_host_template<MmioDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, NONE, base + "mmio/" + end);
}

void run_memset_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    memset_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, base + "ddr/" + end);
    memset_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, base + "hbm/" + end);
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
    // read_test_host_template<MmioDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, NONE, base + "mmio/" + end);
}

void run_copy_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, HOST_ID, WRITE, base + "ddr_ddr/" + end);
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, DEVICE_ID, WRITE, base + "ddr_hbm/" + end);
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, DEVICE_ID, WRITE, base + "hbm_hbm/" + end);
    copy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, HOST_ID, WRITE, base + "hbm_ddr/" + end);
}

void run_memcpy_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    memcpy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, HOST_ID, WRITE, base + "ddr_ddr/" + end);
    memcpy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, HOST_ID, WRITE, DEVICE_ID, WRITE, base + "ddr_hbm/" + end);
    memcpy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, DEVICE_ID, WRITE, base + "hbm_hbm/" + end);
    memcpy_test_host_template<MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, n_threads, DEVICE_ID, WRITE, HOST_ID, WRITE, base + "hbm_ddr/" + end);
}