#pragma once

#include "measurement.hpp"
#include "memory.hpp"
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
        uint64_t *data = a + offset; //pointer to a uint64_t, the address to a memory region(like:where is the actual data??), but &data(it is uint64 **) is the adrress of the pointer data itself (like where is the pointer that says where is my actual data??)
        #pragma unroll
        for (size_t j = 0; j < INNER; ++j) {
            uint64_t v;
            // Cache Operators for Memory Load Instructions
            // .ca Cache at all levels, likely to be accessed again.
            // .cg Cache at global level (cache in L2 and below, not L1).
            // .cs Cache streaming, likely to be accessed once.
            // .cv Cache as volatile (consider cached system memory lines stale, fetch again).
        	asm volatile("ld.ca.u64 %0, [%1];" : "=l"(v) : "l"(data)); //inline ptx assembly, load from address data the value into variable v
            *((uint64_t *) &data) ^= v; // fake update,  bitwise XOR between the contents of the pointer with address &data and v
					// (uint64_t *) &data: take the address of the data pointer itself and pretend that that memory region is a uint64 value, *((uint64_t *) &data):read the value that is stored in that uint64 
            tmp = v;
        }
    }
    // dummy work
    offset += tmp;
    if (offset != index) {
        *a = offset;
    }
}
//i need the dummy work so that the compiler wont eliminate parts of the program that doesnt affect execution i need them



void test_cache() {
	std::cout << "started cache test " << std::endl;
    CudaMallocDataFactory factory(sizeof(uint64_t) * BLOCKSIZE * GRIDSIZE); 
    cudaMemset(factory.data, 0x00, sizeof(uint64_t) * BLOCKSIZE * GRIDSIZE); //initiallization of gpu memory to 0, factory.data is a pointer to the start of the memory region
    void *args[] = {(void *) &factory.data}; 
    time_kernel_execution((void *) cache_kernel, GRIDSIZE, BLOCKSIZE, args, 0, 0); // warmup, i might have memory migration etc
    double time = time_kernel_execution((void *) cache_kernel, GRIDSIZE, BLOCKSIZE, args, 0, 0); //measure cache time kernel, essentially measuring how much time caching on all caches takes
    std::cout << time << std::endl;
    double read_bytes = sizeof(uint64_t) * OUTER * INNER * BLOCKSIZE * GRIDSIZE; 
    std::cout << read_bytes / ((time * 1000000.)) << std::endl; //bytes per sec = throughput GB/sec
	std::cout << "cache test ended" << std::endl;
}
