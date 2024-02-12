#pragma once

#include "measurement.hpp"
#include "data.hpp"
#include "thread_tools.hpp"
#include <iostream>

#define OUTER (8192)
#define INNER (64)
#define BLOCKSIZE (1024)
#define GRIDSIZE (264)

__global__ void cache_kernel(uint64_t *a) {
    int index = BLOCKSIZE * blockIdx.x + threadIdx.x;

    int offset = index;
    int tmp = 0;
    for (size_t i = 0; i < OUTER; ++i) {
        offset += tmp; // fake update
        uint64_t *data = a + offset;
        #pragma unroll
        for (size_t j = 0; j < INNER; ++j) {
            uint64_t v;
            // Cache Operators for Memory Load Instructions
            // .ca Cache at all levels, likely to be accessed again.
            // .cg Cache at global level (cache in L2 and below, not L1).
            // .cs Cache streaming, likely to be accessed once.
            // .cv Cache as volatile (consider cached system memory lines stale, fetch again).
        	asm volatile("ld.ca.u64 %0, [%1];" : "=l"(v) : "l"(data));
            *((uint64_t *) &data) ^= v; // fake update
            tmp = v;
        }
    }
    // dummy work
    offset += tmp;
    if (offset != index) {
        *a = offset;
    }
}


void test_cache() {
    CudaMallocDataFactory factory(sizeof(uint64_t) * BLOCKSIZE * GRIDSIZE);
    cudaMemset(factory.data, 0x00, sizeof(uint64_t) * BLOCKSIZE * GRIDSIZE);
    void *args[] = {(void *) &factory.data}; 
    time_kernel_execution((void *) cache_kernel, GRIDSIZE, BLOCKSIZE, args, 0, 0); // warmup
    double time = time_kernel_execution((void *) cache_kernel, GRIDSIZE, BLOCKSIZE, args, 0, 0);
    std::cout << time << std::endl;
    double read_bytes = sizeof(uint64_t) * OUTER * INNER * BLOCKSIZE * GRIDSIZE;
    std::cout << read_bytes / ((time * 1000000.)) << std::endl;
}