#pragma once

#include "read_write_copy.hpp"

uint64_t host_sweep_test(size_t base_threads, uint8_t *host_data, uint8_t *device_data, size_t n_elems_host, size_t n_elems_device, uint64_t *end_times) {
    size_t host_per_thread_n_cachelines = (n_elems_host / 8) / base_threads;
    size_t device_per_thread_n_cachelines = (n_elems_device / 8) / (72 - base_threads);
    uint64_t nominal_start_time = get_cpu_clock() + 1000000; // cur + 1ms
    for (size_t i = 0; i < 72; ++i) {
        thread_data *cur = &thread_array[i];
        cur->command = READ_TEST;
        cur->start_time = nominal_start_time;
        if (i < base_threads) {
            cur->size = host_per_thread_n_cachelines * 8;
            cur->buffer = host_data;
            host_data += cur->size * sizeof(double);
        } else {
            cur->size = device_per_thread_n_cachelines * 8;
            cur->buffer = device_data;
            device_data += cur->size * sizeof(double);    
        }
    }

    for (size_t i = 0; i < 72; ++i) {
        thread_data *cur = &thread_array[i];
        cur->rx_mutex.lock();
        cur->tx_mutex.unlock(); // send command
    }

    for (size_t i = 0; i < 72; ++i) {
        thread_data *cur = &thread_array[i];
        cur->rx_mutex.lock(); // wait until thread is done
        cur->rx_mutex.unlock();
        end_times[i] = cur->end_time;
    }

    return nominal_start_time;
}

double host_sweep_single(uint8_t *host_data, uint8_t *device_data, size_t n_elems_host, size_t n_elems_device, double ratio, size_t base_threads) {
    uint64_t end_times[72];

    uint64_t nominal_start_time = host_sweep_test(base_threads, host_data, device_data, n_elems_host, n_elems_device, end_times);

    uint64_t max_end = 0;
    for (size_t i = 0; i < 72; ++i) {
        max_end = std::max(max_end, end_times[i]);
    }

    return get_elapsed_milliseconds_clock(nominal_start_time, max_end);
}

void run_host_sweep_tests(size_t n_iter, size_t n_bytes, double ratio, size_t base_threads) {
    double times[n_iter];
    size_t host_bytes = n_bytes * ratio;
    size_t device_bytes = n_bytes - host_bytes;
    HOST_MEM host(host_bytes);
    DEVICE_MEM device(device_bytes);

    for (size_t i = 0; i < n_iter; ++i) {
        prepare_memory(WRITE, host.data, host_bytes);
        prepare_memory(WRITE, device.data, device_bytes);
        prepare_memory(READ, cache_filler, sizeof(cache_filler));
        times[i] = host_sweep_single(host.data, device.data, host_bytes / sizeof(double), device_bytes / sizeof(double), ratio, base_threads);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/sweep/host/" + std::to_string(base_threads) + "/" + std::to_string(ratio));
}

__global__ void device_read_kernel_sync_sweep(double *host_data, double *device_data, size_t base_blocks, size_t n_elems_host, size_t n_elems_device, size_t n_iter, clock_t *time, clock_t *sync) {
    uint64_t dummy;
    __shared__ clock_t clocks[1024];

    size_t actual_block_idx = blockIdx.x < base_blocks ? blockIdx.x : blockIdx.x - base_blocks;
    size_t tid = threadIdx.x + blockDim.x * actual_block_idx;

    auto grid = cg::this_grid();

    double *data = blockIdx.x < base_blocks ? host_data : device_data;
    size_t n_elems = blockIdx.x < base_blocks ? n_elems_host : n_elems_device;
    size_t stride = blockDim.x * (blockIdx.x < base_blocks ? base_blocks : 264 - base_blocks);

    for (size_t iter = 0; iter < n_iter; ++iter) {
        // SYNC -------------------
        grid.sync();
        *sync = get_gpu_clock() + (1000000/32)*32;
        grid.sync();
        __syncthreads();
        clock_t start = *sync;
        while (get_gpu_clock() < start) {}
        // ------------------------

        // for (size_t outer = 0; outer < 1024; ++outer) {
            for (size_t i = tid; i < n_elems; i += stride) {
                uint64_t v;
                asm volatile("ld.ca.u64 %0, [%1];" : "=l"(v) : "l"(&data[i]));
                dummy ^= v;
            }
        // }

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void run_device_sweep_tests(size_t n_iter, size_t n_bytes, size_t base_blocks, double ratio) {
    // int numBlocksPerSm;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, device_read_kernel_sync_sweep, 1024, 0);
    // std::cout << numBlocksPerSm << std::endl;
    size_t host_bytes = n_bytes * ratio;
    size_t device_bytes = n_bytes - host_bytes;
    HOST_MEM host(host_bytes);
    DEVICE_MEM device(device_bytes);
    size_t n_elems_host = host_bytes / sizeof(double);
    size_t n_elems_device = device_bytes / sizeof(double);

    clock_t *big_times = (clock_t *) alloca(sizeof(clock_t) * n_iter * 264);
    clock_t small_times[n_iter];
    clock_t sync;
    clock_t *sync_ptr = &sync;
    // void *args[] = {(void *) &device.data, (void *) &n_elems_device, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
    // cudaLaunchCooperativeKernel((void *) device_read_kernel_sync, 264, 1024, args, 0, 0);
    void *args[] = {(void *) &host.data, (void *) &device.data, (void *) &base_blocks, (void *) &n_elems_host, (void *) &n_elems_device, (void *) &n_iter, (void *) &big_times, (void *) &sync_ptr};
    gpuErrchk(cudaLaunchCooperativeKernel((void *) device_read_kernel_sync_sweep, 264, 1024, args, 0, 0));
    gpuErrchk(cudaDeviceSynchronize());
    for (size_t itr = 0; itr < n_iter; ++itr) {
        clock_t t = big_times[itr * 264];
        for (size_t block_id = 1; block_id < 264; ++block_id) {
            t = max(t, big_times[itr * 264 + block_id]);
        }
        small_times[itr] = t;
    }
    times_to_file(small_times, n_iter, n_bytes, "results/sweep/device/" + std::to_string(base_blocks) + "/" + std::to_string(ratio), 1000000000.);
}
