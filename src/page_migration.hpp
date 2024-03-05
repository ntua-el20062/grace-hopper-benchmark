#include "memory.hpp"
#include "thread_tools.hpp"
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void page_migration_kernel(double *data, size_t n_bytes, size_t n_iterations, clock_t *start_times, clock_t *end_times) {
    double dummy[2];
    __shared__ clock_t start_vector[1024];
    __shared__ clock_t end_vector[1024];

    auto grid = cg::this_thread_block();

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t iter = 0; iter < n_iterations; ++iter) {
        // enforce a common starting point - should be equal to launching a new kernel, but with less overhead
        //grid.sync();
       grid.sync();

        // run the iteration
        clock_t start = clock();
        for (size_t i = tid; i < n_bytes / sizeof(double); i += blockDim.x * gridDim.x) {
            dummy[i%2] = data[i];
        }
        clock_t end = clock();

        // store the start and end clock in shared memory
        start_vector[threadIdx.x] = start;
        end_vector[threadIdx.x] = end;

        // make sure all threads in the block have saved the clocks
       grid.sync();

        if (threadIdx.x == 0) {
            // reduce at the block level
            for (size_t i = 0; i < blockDim.x; ++i) {
                start = min(start, start_vector[i]);
                end = max(end, end_vector[i]);
            }
            // save globally the block level start and end clocks
            start_times[blockIdx.x + iter * gridDim.x] = start;
            end_times[blockIdx.x + iter * gridDim.x] = end;
        }


        // dummy print of element
        if (tid == -1) {
            printf("%lf ", dummy[0]);
        }
    }
}

__global__ void page_migration_kernel_block(double *data, size_t n_bytes, size_t n_iterations, clock_t *times) {
    double dummy[2];
    __shared__ clock_t start_vector[1024];
    __shared__ clock_t end_vector[1024];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t iter = 0; iter < n_iterations; ++iter) {
        if (tid == 0) {
            start_vector[0] = 0;
        }
        __syncthreads();
        KERNEL_SYNC(start_vector);
        for (size_t i = tid; i < n_bytes / sizeof(double); i += blockDim.x * gridDim.x) {
            dummy[i%2] = data[i];
        }
        KERNEL_MEASURE(start_vector);
        __syncthreads();
        if (threadIdx.x == 0) {
            clock_t elapsed = start_vector[0];
            for (size_t i = 1; i < blockDim.x; ++i) {
                elapsed = max(elapsed, start_vector[i]);
            }
            times[iter * gridDim.x + blockIdx.x] = elapsed;
        }
        // dummy print of element
        if (tid == -1) {
            printf("%lf ", dummy[0]);
        }
    }
}

__global__ void page_migration_kernel_simple(double *data, size_t n_bytes) {
    double dummy[2];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (size_t i = tid; i < n_bytes / sizeof(double); i += blockDim.x * gridDim.x) {
        dummy[i%2] = data[i];
    }
    if (tid == -1) {
        printf("%lf ", dummy[0]);
    }
}