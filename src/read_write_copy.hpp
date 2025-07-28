#pragma once

#include "memory.hpp"
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

__global__ void device_read_kernel_sync(ulonglong2 *data, size_t n_elems, size_t n_iter, clock_t *time, volatile clock_t *sync) {
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

        for (size_t i = tid; i < n_elems / 2; i += 1024 * gridDim.x) {
            ulonglong2 cur = data[i];
            dummy ^= cur.x ^ cur.y;
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

__global__ void infinite_device_read_kernel(ulonglong2 *data, size_t n_elems, volatile clock_t *sync) {
    uint64_t dummy;

    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    while (true) {
        for (size_t i = tid; i < n_elems / 2; i += 1024 * 264) {
            ulonglong2 cur = data[i];
            dummy ^= cur.x ^ cur.y;
        }
        if (*sync == 1) {
            printf("%lu ", dummy);
        }
    }
}

__global__ void device_write_kernel_sync(uint64_t *a, size_t n_elems, size_t n_iter, clock_t *time, clock_t *sync) {
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
     std::cout << "start run_clock_analysis_host" << std::endl;
     size_t n_threads = 72;
     size_t n_iter = 10;
     uint64_t start_times[n_iter * n_threads];
     uint64_t end_times[n_iter * n_threads];
     size_t n_bytes = 1UL << 32;
     size_t n_elems = n_bytes / sizeof(double);
     MmapDataFactory factory(n_bytes);
     dispatch_command(HOST_ID, WRITE, factory.data, n_bytes);
     for (size_t itr = 0; itr < n_iter; itr++) {
	 std::cout << "HIIIIIIIIIIIIIIIIIIIII" << std::endl;
         uint64_t nominal_time = run_test(WRITE_TEST, n_threads, 0, factory.data, nullptr, n_elems, &start_times[itr * n_threads], &end_times[itr * n_threads]);
         for (size_t j = 0; j < n_threads; ++j) {
             start_times[itr * n_threads + j] -= nominal_time;
             end_times[itr * n_threads + j] -= nominal_time;
         }
     }
     for (size_t itr = 0; itr < n_iter; itr++) {
	             std::filesystem::create_directories(std::filesystem::path("results/clock_analysis/host/start/").parent_path());
                     std::filesystem::create_directories(std::filesystem::path("results/clock_analysis/host/end/").parent_path());

         std::ofstream start_file("results/clock_analysis/host/start/" + std::to_string(itr));
         std::ofstream end_file("results/clock_analysis/host/end/" + std::to_string(itr));
         for (size_t i = 0; i < n_threads; i++) {
             start_file << start_times[itr * n_threads + i] << std::endl;
             end_file << end_times[itr * n_threads + i] << std::endl;
         }
     }
     std::cout << "done run_clock_analysis_host" << std::endl;

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

__global__ void device_read_kernel_block(ulonglong2 *data, size_t n_elems, size_t n_iter, volatile clock_t *time) {
    uint64_t dummy;
    __shared__ clock_t clocks[1024];

    size_t tid = threadIdx.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        if (tid == 0) *clocks = clock() + (19800000/7)*7;
        __syncthreads();
        clock_t start = *clocks;
        while (clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems / 2; i += 1024) {
            ulonglong2 cur = data[i];
            dummy ^= cur.x ^ cur.y;
        }

        clock_t end = clock();
        clocks[threadIdx.x] = end;

        // STORE -----------------
        __syncthreads();
        if (tid == 0) {
            clock_t t = clocks[0];
            for (size_t i = 1; i < 1024; ++i) {
                t = max(t, clocks[i]);
            }
            time[iter] = (t - start);
        }
        // ----------------------
    }
    if (*time == 0) {
        printf("%lu ", dummy);
    }
}

__global__ void device_write_kernel_block(double *a, size_t n_elems, size_t n_iter, clock_t *time) {
    __shared__ clock_t clocks[1024];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        __syncthreads();
        if (tid == 0) *clocks = clock() + (19800000/7)*7;
        __syncthreads();
        clock_t start = *clocks;
        while (clock() < start) {}
        // ------------------------

        for (size_t i = tid; i < n_elems; i += 1024) {
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
void write_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, int device, std::string name) {
    size_t n_elems = n_bytes / sizeof(double);
    cudaSetDevice(device);
    ALLOC factory(n_bytes);
    if (grid_size == 0) { // run with only one thread
        clock_t times[n_iter];
        device_write_kernel_single<<<1, 1>>>((double *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/write/device/single/" + name);
    // } else if (grid_size == 1) { // run with only one block
    //     clock_t times[n_iter];
    //     device_write_kernel_block<<<1, 1024>>>((double *) factory.data, n_elems, n_iter, times);
    //     cudaDeviceSynchronize();
    //     times_to_file(times, n_iter, n_bytes, "results/write/device/" + name);
    } else {
        clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * grid_size);
        clock_t small_times[n_iter];
        clock_t sync;
        clock_t *sync_ptr = &sync;
        void *args[] = {(void *) &factory.data, (void *) &n_elems, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
        cudaLaunchCooperativeKernel((void *) device_write_kernel_sync, grid_size, 1024, args, 0, 0);
        cudaDeviceSynchronize();
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/write/device/" + name, 1000000000.);
    }
    cudaSetDevice(0);
}

template <typename ALLOC>
void read_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, int device, std::string name) {
    size_t n_elems = n_bytes / sizeof(double);
    cudaSetDevice(device);
    ALLOC factory(n_bytes);
    // cudaMemPrefetchAsync(factory.data, n_bytes, 0, 0);
    if (grid_size == 0) { // run with only one thread
        clock_t times[n_iter];
        device_read_kernel_single<<<1, 1>>>((uint64_t *) factory.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/read/device/single/" + name);
    // } else if (grid_size == 1) { // run with only one block
    //     clock_t times[n_iter];
    //     cudaMemset(factory.data, 0x00, n_bytes);
    //     device_read_kernel_block<<<1, 1024>>>((ulonglong2 *) factory.data, n_elems, n_iter, times);
    //     cudaDeviceSynchronize();
    //     times_to_file(times, n_iter, n_bytes, "results/read/device/" + name);
    } else {
        clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * grid_size);
        clock_t small_times[n_iter];
        clock_t sync;
        clock_t *sync_ptr = &sync;
        void *args[] = {(void *) &factory.data, (void *) &n_elems, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
        cudaLaunchCooperativeKernel((void *) device_read_kernel_sync, grid_size, 1024, args, 0, 0);
        cudaDeviceSynchronize();
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/read/device/" + name, 1000000000.);
    }
    cudaSetDevice(0);
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void copy_test_device_template(size_t n_iter, size_t n_bytes, int grid_size, std::string name) {
    //double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);

    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);

    if (grid_size == 0) { // run with only one thread
        clock_t times[n_iter];
        device_copy_kernel_single<<<1, 1>>>((double *) src.data, (double *) dst.data, n_elems, n_iter, times);
        cudaDeviceSynchronize();
        times_to_file(times, n_iter, n_bytes, "results/copy/device/single/" + name);
    // } else if (grid_size == 1) { // run with only one block
    //     clock_t times[n_iter];
    //     device_copy_kernel_block<<<1, 1024>>>((double *) src.data, (double *) dst.data, n_elems, n_iter, times);
    //     cudaDeviceSynchronize();
    //     times_to_file(times, n_iter, n_bytes, "results/copy/device/" + name);
    } else {
        clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * grid_size);
        clock_t small_times[n_iter];
        clock_t sync;
        clock_t *sync_ptr = &sync;
        void *args[] = {(void *) &src.data, (void *) &dst.data, (void *) &n_elems, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
        cudaLaunchCooperativeKernel((void *) device_copy_kernel_sync, grid_size, 1024, args, 0, 0);
        cudaDeviceSynchronize();
        for (size_t itr = 0; itr < n_iter; ++itr) {
            clock_t t = big_times[itr * grid_size];
            for (size_t block_id = 1; block_id < grid_size; ++block_id) {
                t = max(t, big_times[itr * grid_size + block_id]);
            }
            small_times[itr] = t;
        }
        times_to_file(small_times, n_iter, n_bytes, "results/copy/device/" + name, 1000000000.);
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
    write_test_device_template<HOST_MEM>(n_iter, n_bytes, grid_size, 0, base + "ddr/" + end);
    write_test_device_template<DEVICE_MEM>(n_iter, n_bytes, grid_size, 0, base + "hbm/" + end);
    //write_test_device_template<REMOTE_HOST_MEM>(n_iter, n_bytes, grid_size, 0, base + "ddr_remote/" + end);
    //write_test_device_template<REMOTE_DEVICE_MEM>(n_iter, n_bytes, grid_size, 0, base + "hbm_remote/" + end);
}

void run_read_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
    read_test_device_template<HOST_MEM>(n_iter, n_bytes, grid_size, 0, base + "ddr/" + end);
    read_test_device_template<DEVICE_MEM>(n_iter, n_bytes, grid_size, 0, base + "hbm/" + end);
    //read_test_device_template<REMOTE_HOST_MEM>(n_iter, n_bytes, grid_size, 0, base + "ddr_remote/" + end);
    //read_test_device_template<REMOTE_DEVICE_MEM>(n_iter, n_bytes, grid_size, 0, base + "hbm_remote/" + end);
}

void run_copy_tests_device(size_t n_iter, size_t n_bytes, int grid_size, std::string base, std::string end) {
	std::cout << "started  run_copy_tests_device" << std::endl;
	copy_test_device_template<HOST_MEM, HOST_MEM>(n_iter, n_bytes, grid_size, base + "ddr_ddr/" + end);
        std::cout << "started  run_copy_tests_device host host" << std::endl;

	copy_test_device_template<HOST_MEM, DEVICE_MEM>(n_iter, n_bytes, grid_size, base + "ddr_hbm/" + end);
        std::cout << " started copy_test_device_template host dev" << std::endl;
    
	copy_test_device_template<DEVICE_MEM, DEVICE_MEM>(n_iter, n_bytes, grid_size, base + "hbm_hbm/" + end);
        std::cout << "started   copy_test_device_template dev dev" << std::endl;
    
	copy_test_device_template<DEVICE_MEM, HOST_MEM>(n_iter, n_bytes, grid_size, base + "hbm_ddr/" + end);
        std::cout << "done  run_copy_tests_device dev host" << std::endl;

}

// ---------------- HOST TEMPLATES --------------------------------
template <typename ALLOC>
void write_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t initial_thread, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(WRITE_TEST, n_threads, initial_thread, factory.data, nullptr, n_elems);
    }
    size_t per_thread_n_cachelines = (n_bytes / 64) / n_threads;
    size_t actual_bytes = per_thread_n_cachelines * 64 * n_threads;
    millisecond_times_to_gb_sec_file(times, n_iter, actual_bytes, "results/write/host/" + name);
}

template <typename ALLOC>
void memset_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t initial_thread, size_t target, ThreadCommand command, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_command(target, command, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(MEMSET_TEST, n_threads, initial_thread, factory.data, nullptr, n_elems);
    }
    size_t per_thread_n_cachelines = (n_bytes / 64) / n_threads;
    size_t actual_bytes = per_thread_n_cachelines * 64 * n_threads;
    millisecond_times_to_gb_sec_file(times, n_iter, actual_bytes, "results/memset/host/" + name);
}

template <typename ALLOC>
void read_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t initial_thread, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);
    ALLOC factory(n_bytes);
    dispatch_command(initial_thread, WRITE, factory.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(READ_TEST, n_threads, initial_thread, factory.data, nullptr, n_elems) / 1;
    }
    size_t per_thread_n_cachelines = (n_bytes / 64) / n_threads;
    size_t actual_bytes = per_thread_n_cachelines * 64 * n_threads;
    millisecond_times_to_gb_sec_file(times, n_iter, actual_bytes, "results/read/host/" + name);
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void copy_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t initial_thread, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);

    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(COPY_TEST, n_threads, initial_thread, src.data, dst.data, n_elems);
    }
    size_t per_thread_n_cachelines = (n_bytes / 64) / n_threads;
    size_t actual_bytes = per_thread_n_cachelines * 64 * n_threads;
    millisecond_times_to_gb_sec_file(times, n_iter, actual_bytes, "results/copy/host/" + name);
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void memcpy_test_host_template(size_t n_iter, size_t n_bytes, size_t n_threads, size_t initial_thread, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);

    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);
    dispatch_command(src_target, src_mode, src.data, n_bytes);
    dispatch_command(dst_target, dst_mode, dst.data, n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(MEMCPY_TEST, n_threads, initial_thread, src.data, dst.data, n_elems);
    }
    size_t per_thread_n_cachelines = (n_bytes / 64) / n_threads;
    size_t actual_bytes = per_thread_n_cachelines * 64 * n_threads;
    millisecond_times_to_gb_sec_file(times, n_iter, actual_bytes, "results/memcpy/host/" + name);
}

// ---------------- HOST TESTS --------------------------------
void run_write_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    //write_test_host_template<HOST_MEM>(n_iter, n_bytes, n_threads, 0, base + "ddr/" + end); //-----------------------------------------------------------to thelw kai ayto, apla to exw hdh kanei!!!!
    write_test_host_template<DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, base + "hbm/" + end); //-------------------------------------------------------to thelw kai ayto, apla to exw hdh kanei!!!!
    //write_test_host_template<REMOTE_HOST_MEM>(n_iter, n_bytes, n_threads, 0, base + "ddr_remote/" + end);
    //write_test_host_template<REMOTE_DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, base + "hbm_remote/" + end);
    // write_test_host_template<FAR_HOST_MEM>(n_iter, n_bytes, n_threads, 72, base + "ddr_far/" + end);
    // write_test_host_template<FAR_DEVICE_MEM>(n_iter, n_bytes, n_threads, 72, base + "hbm_far/" + end);
}

void run_memset_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    memset_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, 0, HOST_ID, WRITE, base + "ddr/" + end);
    memset_test_host_template<MmapDataFactory>(n_iter, n_bytes, n_threads, 0, DEVICE_ID, WRITE, base + "hbm/" + end);
}

void run_read_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    read_test_host_template<HOST_MEM>(n_iter, n_bytes, n_threads, 0, base + "ddr/" + end);
    read_test_host_template<DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, base + "hbm/" + end);
    //read_test_host_template<REMOTE_HOST_MEM>(n_iter, n_bytes, n_threads, 0, base + "ddr_remote/" + end);
    //read_test_host_template<REMOTE_DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, base + "hbm_remote/" + end);
}

void run_copy_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
    copy_test_host_template<HOST_MEM, HOST_MEM>(n_iter, n_bytes, n_threads, 0, base + "ddr_ddr/" + end);
    copy_test_host_template<HOST_MEM, DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, base + "ddr_hbm/" + end);
    copy_test_host_template<DEVICE_MEM, DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, base + "hbm_hbm/" + end);
    copy_test_host_template<DEVICE_MEM, HOST_MEM>(n_iter, n_bytes, n_threads, 0, base + "hbm_ddr/" + end);
}

 void run_memcpy_tests_host(size_t n_iter, size_t n_bytes, size_t n_threads, std::string base, std::string end) {
     memcpy_test_host_template<HOST_MEM, HOST_MEM>(n_iter, n_bytes, n_threads, 0, HOST_ID, WRITE, HOST_ID, WRITE, base + "ddr_ddr/" + end);
     memcpy_test_host_template<HOST_MEM, DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, HOST_ID, WRITE,DEVICE_ID, WRITE, base + "ddr_hbm/" + end);
     memcpy_test_host_template<DEVICE_MEM, DEVICE_MEM>(n_iter, n_bytes, n_threads, 0, DEVICE_ID, WRITE, DEVICE_ID, WRITE, base + "hbm_hbm/" + end);
     memcpy_test_host_template<DEVICE_MEM, HOST_MEM>(n_iter, n_bytes, n_threads, 0, DEVICE_ID, WRITE, HOST_ID, WRITE, base + "hbm_ddr/" + end);
 }


// ---------------- CUDAMEMCPY TESTS --------------------------------
template <typename SRC_ALLOC, typename DST_ALLOC>
void cuda_memcpy_template(size_t n_iter, size_t n_bytes, std::string name) {
    double times[n_iter];

    // if constexpr (SRC_ALLOC::is_gpu && DST_ALLOC::is_gpu) {
    //     std::cout << DST_ALLOC::gpu_id << " " << SRC_ALLOC::gpu_id << std::endl;
    // }
    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        uint64_t start = get_cpu_clock();
        // if constexpr (SRC_ALLOC::is_gpu && DST_ALLOC::is_gpu) {
        //     cudaMemcpyPeerAsync(dst.data, DST_ALLOC::gpu_id, src.data, SRC_ALLOC::gpu_id, n_bytes);
        // } else {
        // cudaMemcpyAsync(dst.data, src.data, n_bytes, cudaMemcpyDefault);
        cudaMemcpy(dst.data, src.data, n_bytes, cudaMemcpyDefault);
        // }
        cudaDeviceSynchronize();
        uint64_t end = get_cpu_clock();
        times[i] = get_elapsed_milliseconds_clock(start, end);

    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/memcpy/cuda/" + name);
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void device_copy_template(size_t n_iter, size_t n_bytes, std::string name) {
    clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * 264);
    clock_t small_times[n_iter];
    clock_t sync;
    clock_t *sync_ptr = &sync;
    size_t n_elems = n_bytes / sizeof(double);

    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);
    cudaMemset(src.data, 0xff, n_bytes);
    cudaDeviceSynchronize();
    void *args[] = {(void *) &src.data, (void *) &dst.data, (void *) &n_elems, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
    cudaLaunchCooperativeKernel((void *) device_copy_kernel_sync, 264, 1024, args, 0, 0);
    cudaDeviceSynchronize();

    for (size_t itr = 0; itr < n_iter; ++itr) {
        clock_t t = big_times[itr * 264];
        for (size_t block_id = 1; block_id < 264; ++block_id) {
            t = max(t, big_times[itr * 264 + block_id]);
        }
        small_times[itr] = t;
    }

    times_to_file(small_times, n_iter, n_bytes, "results/copy/device/" + name, 1000000000.);
}

template <typename SRC_ALLOC, typename DST_ALLOC>
void host_copy_template(size_t n_iter, size_t n_bytes, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(double);

    SRC_ALLOC src(n_bytes);
    DST_ALLOC dst(n_bytes);
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_test(COPY_TEST, 72, 0, src.data, dst.data, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/copy/host/" + name);
}

void run_cuda_memcpy_heatmap_tests() {
    size_t n_iter = 10;
    size_t n_bytes = 1UL << 33;
    std::cout << "BEGIN HEATMAP TESTS" << std::endl;
    cuda_memcpy_template<CudaMallocDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/CMC_CMC");
    cuda_memcpy_template<CudaMallocDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/CMC_CMH");
    cuda_memcpy_template<CudaMallocDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/CMC_MMH");
    cuda_memcpy_template<CudaMallocDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/CMC_MMD");
    cuda_memcpy_template<CudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMC_MGH");
    cuda_memcpy_template<CudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMC_MGD");
    //cuda_memcpy_template<CudaMallocDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/CMC_CMR");
    //cuda_memcpy_template<CudaMallocDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap1/CMC_MDR");
    //cuda_memcpy_template<CudaMallocDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/CMC_MHR");
    std::cout << "DONE WITH FIRST SET" << std::endl;

    cuda_memcpy_template<CudaMallocHostDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/CMH_CMC");
    cuda_memcpy_template<CudaMallocHostDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/CMH_CMH");
    cuda_memcpy_template<CudaMallocHostDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/CMH_MMH");
    cuda_memcpy_template<CudaMallocHostDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/CMH_MMD");
    cuda_memcpy_template<CudaMallocHostDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMH_MGH");
    cuda_memcpy_template<CudaMallocHostDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMH_MGD");
    //cuda_memcpy_template<CudaMallocHostDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/CMH_CMR");
    //cuda_memcpy_template<CudaMallocHostDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/CMH_MDR");
    //cuda_memcpy_template<CudaMallocHostDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/CMH_MHR");

    cuda_memcpy_template<NumaDataFactory<0>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MMH_CMC");
    cuda_memcpy_template<NumaDataFactory<0>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MMH_CMH");
    cuda_memcpy_template<NumaDataFactory<0>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MMH_MMH");
    cuda_memcpy_template<NumaDataFactory<0>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MMH_MMD");
    cuda_memcpy_template<NumaDataFactory<0>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMH_MGH");
    cuda_memcpy_template<NumaDataFactory<0>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMH_MGD");
    //cuda_memcpy_template<NumaDataFactory<0>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/MMH_CMR");
    //cuda_memcpy_template<NumaDataFactory<0>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MMH_MDR");
    //cuda_memcpy_template<NumaDataFactory<0>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MMH_MHR");

    cuda_memcpy_template<NumaDataFactory<1>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MMD_CMC");
    cuda_memcpy_template<NumaDataFactory<1>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MMD_CMH");
    cuda_memcpy_template<NumaDataFactory<1>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MMD_MMH");
    cuda_memcpy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MMD_MMD");
    cuda_memcpy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMD_MGH");
    cuda_memcpy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMD_MGD");
    //cuda_memcpy_template<NumaDataFactory<1>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/MMD_CMR");
    //cuda_memcpy_template<NumaDataFactory<4>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MMD_MDR");
    //cuda_memcpy_template<NumaDataFactory<4>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MMD_MHR");

    cuda_memcpy_template<ManagedMemoryDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MGH_CMC");
    cuda_memcpy_template<ManagedMemoryDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MGH_CMH");
    cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MGH_MMH");
    cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MGH_MMD");
    cuda_memcpy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGH_MGH");
    cuda_memcpy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGH_MGD");
    //cuda_memcpy_template<ManagedMemoryDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/MGH_CMR");
    //cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MGH_MDR");
    //cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MGH_MHR");

    cuda_memcpy_template<ManagedMemoryDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MGD_CMC");
    cuda_memcpy_template<ManagedMemoryDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MGD_CMH");
    cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MGD_MMH");
    cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MGD_MMD");
    cuda_memcpy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGD_MGH");
    cuda_memcpy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGD_MGD");
    //cuda_memcpy_template<ManagedMemoryDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/MGD_CMR");
    //cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MGD_MDR");
    //cuda_memcpy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MGD_MHR");

    //-----------------------------------------------------------------------------------------------------------------REMOTE----------------------------------------------------------------------------
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/CMR_CMC");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/CMR_CMH");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/CMR_MMH");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/CMR_MMD");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMR_MGH");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMR_MGD");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/CMR_CMR");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/CMR_MDR");
    //cuda_memcpy_template<RemoteCudaMallocDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/CMR_MHR");

    //cuda_memcpy_template<NumaDataFactory<12>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap/MDR_CMC");
    //cuda_memcpy_template<NumaDataFactory<12>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap/MDR_CMH");
    //cuda_memcpy_template<NumaDataFactory<12>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap/MDR_MMH");
    /*//cuda_memcpy_template<NumaDataFactory<12>, NumaDataFactory<4>>            (n_iter, n_bytes, "heatmap/MDR_MMD");
    //cuda_memcpy_template<NumaDataFactory<12>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MDR_MGH");
    //cuda_memcpy_template<NumaDataFactory<12>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MDR_MGD");
    //cuda_memcpy_template<NumaDataFactory<12>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/MDR_CMR");
    //cuda_memcpy_template<NumaDataFactory<12>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MDR_MDR");
    //cuda_memcpy_template<NumaDataFactory<12>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MDR_MHR");
    */
    //cuda_memcpy_template<NumaDataFactory<1>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap/MHR_CMC");
    //cuda_memcpy_template<NumaDataFactory<1>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap/MHR_CMH");
    //cuda_memcpy_template<NumaDataFactory<1>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap/MHR_MMH");
    //cuda_memcpy_template<NumaDataFactory<1>, NumaDataFactory<4>>            (n_iter, n_bytes, "heatmap/MHR_MMD");
    //cuda_memcpy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MHR_MGH");
    //cuda_memcpy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MHR_MGD");
    //cuda_memcpy_template<NumaDataFactory<1>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/MHR_CMR");
    //cuda_memcpy_template<NumaDataFactory<1>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MHR_MDR");
    //cuda_memcpy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MHR_MHR");
}

void run_device_copy_heatmap_tests() {
    size_t n_iter = 10;
    size_t n_bytes = 1UL << 33;
    std::cout << "began run_device_copy_heatmap_tests()" << std::endl;
    device_copy_template<CudaMallocDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/CMC_CMC");
    device_copy_template<CudaMallocDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/CMC_CMH");
    device_copy_template<CudaMallocDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/CMC_MMH");
        std::cout << "1" << std::endl;

    device_copy_template<CudaMallocDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/CMC_MMD");
    device_copy_template<CudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMC_MGH");
    device_copy_template<CudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMC_MGD");
    //device_copy_template<CudaMallocDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/CMC_CMR");
    //device_copy_template<CudaMallocDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/CMC_MDR");
    //device_copy_template<CudaMallocDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/CMC_MHR");
        std::cout << "2" << std::endl;

    device_copy_template<CudaMallocHostDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/CMH_CMC");
    device_copy_template<CudaMallocHostDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/CMH_CMH");
    device_copy_template<CudaMallocHostDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/CMH_MMH");
    device_copy_template<CudaMallocHostDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/CMH_MMD");
    device_copy_template<CudaMallocHostDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMH_MGH");
    device_copy_template<CudaMallocHostDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMH_MGD");
    //device_copy_template<CudaMallocHostDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/CMH_CMR");
    //device_copy_template<CudaMallocHostDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/CMH_MDR");
    //device_copy_template<CudaMallocHostDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/CMH_MHR");
        std::cout << "3" << std::endl;

    device_copy_template<NumaDataFactory<0>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MMH_CMC");
    device_copy_template<NumaDataFactory<0>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MMH_CMH");
    device_copy_template<NumaDataFactory<0>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MMH_MMH");
    device_copy_template<NumaDataFactory<0>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MMH_MMD");
    device_copy_template<NumaDataFactory<0>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMH_MGH");
    device_copy_template<NumaDataFactory<0>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMH_MGD");
    //device_copy_template<NumaDataFactory<0>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap1/MMH_CMR");
    //device_copy_template<NumaDataFactory<0>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap1/MMH_MDR");
    //device_copy_template<NumaDataFactory<0>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MMH_MHR");
        std::cout << "4" << std::endl;

    device_copy_template<NumaDataFactory<1>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MMD_CMC");
    device_copy_template<NumaDataFactory<1>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MMD_CMH");
    device_copy_template<NumaDataFactory<1>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MMD_MMH");
    device_copy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MMD_MMD");
    device_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMD_MGH");
    device_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMD_MGD");
    //device_copy_template<NumaDataFactory<1>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/MMD_CMR");
    //device_copy_template<NumaDataFactory<1>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MMD_MDR");
    //device_copy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MMD_MHR");

    device_copy_template<ManagedMemoryDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MGH_CMC");
    device_copy_template<ManagedMemoryDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MGH_CMH");
    device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MGH_MMH");
    device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MGH_MMD");
    device_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGH_MGH");
    device_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGH_MGD");
    //device_copy_template<ManagedMemoryDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/MGH_CMR");
    //device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MGH_MDR");
    //device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MGH_MHR");
        std::cout << "5" << std::endl;

    device_copy_template<ManagedMemoryDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap1/MGD_CMC");
    device_copy_template<ManagedMemoryDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MGD_CMH");
    device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MGD_MMH");
    //device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MGD_MMD");
    device_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGD_MGH");
    device_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGD_MGD");
    //device_copy_template<ManagedMemoryDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/MGD_CMR");
    //device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MGD_MDR");
    //device_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MGD_MHR");
        std::cout << "6" << std::endl;

    //device_copy_template<RemoteCudaMallocDataFactory, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap/CMR_CMC");
    //device_copy_template<RemoteCudaMallocDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap/CMR_CMH");
    //device_copy_template<RemoteCudaMallocDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap/CMR_MMH");
    //device_copy_template<RemoteCudaMallocDataFactory, NumaDataFactory<4>>            (n_iter, n_bytes, "heatmap/CMR_MMD");
    //device_copy_template<RemoteCudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/CMR_MGH");
    //device_copy_template<RemoteCudaMallocDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/CMR_MGD");
    //device_copy_template<RemoteCudaMallocDataFactory, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/CMR_CMR");
    //device_copy_template<RemoteCudaMallocDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/CMR_MDR");
    //device_copy_template<RemoteCudaMallocDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/CMR_MHR");
        std::cout << "7" << std::endl;

    /*
    //device_copy_template<NumaDataFactory<12>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap/MDR_CMC");
    //device_copy_template<NumaDataFactory<12>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap/MDR_CMH");
    //device_copy_template<NumaDataFactory<12>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap/MDR_MMH");
    //device_copy_template<NumaDataFactory<12>, NumaDataFactory<4>>            (n_iter, n_bytes, "heatmap/MDR_MMD");
    //device_copy_template<NumaDataFactory<12>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MDR_MGH");
    //device_copy_template<NumaDataFactory<12>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MDR_MGD");
    //device_copy_template<NumaDataFactory<12>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/MDR_CMR");
    //device_copy_template<NumaDataFactory<12>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MDR_MDR");
    //device_copy_template<NumaDataFactory<12>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MDR_MHR");
    */
    //device_copy_template<NumaDataFactory<1>, CudaMallocDataFactory>      (n_iter, n_bytes, "heatmap/MHR_CMC");
    //device_copy_template<NumaDataFactory<1>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap/MHR_CMH");
    //device_copy_template<NumaDataFactory<1>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap/MHR_MMH");
    //device_copy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MHR_MMD");
    //device_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MHR_MGH");
    //device_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MHR_MGD");
    //device_copy_template<NumaDataFactory<1>, RemoteCudaMallocDataFactory>(n_iter, n_bytes, "heatmap/MHR_CMR");
    //device_copy_template<NumaDataFactory<1>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MHR_MDR");
    //device_copy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MHR_MHR");
        std::cout << "8" << std::endl;

}

void run_host_copy_heatmap_tests() {
    size_t n_iter = 10;
    size_t n_bytes = CEIL(1UL << 33, 72) * 72;
        std::cout << "1" << std::endl;

    host_copy_template<CudaMallocHostDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/CMH_CMH");
    host_copy_template<CudaMallocHostDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/CMH_MMH");
    host_copy_template<CudaMallocHostDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/CMH_MMD");
    host_copy_template<CudaMallocHostDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMH_MGH");
    host_copy_template<CudaMallocHostDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/CMH_MGD");
    //host_copy_template<CudaMallocHostDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/CMH_MDR");
    //host_copy_template<CudaMallocHostDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/CMH_MHR");
        std::cout << "2" << std::endl;

    host_copy_template<NumaDataFactory<0>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MMH_CMH");
    host_copy_template<NumaDataFactory<0>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MMH_MMH");
    host_copy_template<NumaDataFactory<0>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MMH_MMD");
    host_copy_template<NumaDataFactory<0>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMH_MGH");
    host_copy_template<NumaDataFactory<0>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMH_MGD");
    //host_copy_template<NumaDataFactory<0>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MMH_MDR");
    //host_copy_template<NumaDataFactory<0>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MMH_MHR");
        std::cout << "3" << std::endl;

    host_copy_template<NumaDataFactory<1>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MMD_CMH");
    host_copy_template<NumaDataFactory<1>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MMD_MMH");
    host_copy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MMD_MMD");
    host_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMD_MGH");
    host_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MMD_MGD");
    //host_copy_template<NumaDataFactory<1>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MMD_MDR");
    //host_copy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MMD_MHR");

    host_copy_template<ManagedMemoryDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MGH_CMH");
    host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MGH_MMH");
    host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MGH_MMD");
    host_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGH_MGH");
    host_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGH_MGD");
    //host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MGH_MDR");
    //host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MGH_MHR");
        std::cout << "4" << std::endl;

    host_copy_template<ManagedMemoryDataFactory, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap1/MGD_CMH");
    host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap1/MGD_MMH");
    host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap1/MGD_MMD");
    host_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGD_MGH");
    host_copy_template<ManagedMemoryDataFactory, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap1/MGD_MGD");
    //host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MGD_MDR");
    //host_copy_template<ManagedMemoryDataFactory, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MGD_MHR");
        std::cout << "5" << std::endl;

    //host_copy_template<NumaDataFactory<12>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap/MDR_CMH");
    //host_copy_template<NumaDataFactory<12>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap/MDR_MMH");
    //host_copy_template<NumaDataFactory<12>, NumaDataFactory<4>>            (n_iter, n_bytes, "heatmap/MDR_MMD");
    //host_copy_template<NumaDataFactory<12>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MDR_MGH");
    //host_copy_template<NumaDataFactory<12>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MDR_MGD");
    //host_copy_template<NumaDataFactory<12>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MDR_MDR");
    //host_copy_template<NumaDataFactory<12>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MDR_MHR");

    //host_copy_template<NumaDataFactory<1>, CudaMallocHostDataFactory>  (n_iter, n_bytes, "heatmap/MHR_CMH");
    //host_copy_template<NumaDataFactory<1>, NumaDataFactory<0>>            (n_iter, n_bytes, "heatmap/MHR_MMH");
    //host_copy_template<NumaDataFactory<1>, NumaDataFactory<4>>            (n_iter, n_bytes, "heatmap/MHR_MMD");
    //host_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MHR_MGH");
    //host_copy_template<NumaDataFactory<1>, ManagedMemoryDataFactory>   (n_iter, n_bytes, "heatmap/MHR_MGD");
    //host_copy_template<NumaDataFactory<1>, NumaDataFactory<12>>            (n_iter, n_bytes, "heatmap/MHR_MDR");
    //host_copy_template<NumaDataFactory<1>, NumaDataFactory<1>>            (n_iter, n_bytes, "heatmap/MHR_MHR");
        std::cout << "6" << std::endl;

}
