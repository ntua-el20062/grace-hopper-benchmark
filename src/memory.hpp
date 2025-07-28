#pragma once

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <mutex>
#include <thread>
#include "thread_tools.hpp"
#include <algorithm>
#include <random>
#include "random.hpp"
#include <sys/mman.h>
#include <stdlib.h>
#include <fcntl.h>
#include <numa.h>
#include <stdlib.h>
#include <unistd.h>
#include <numaif.h>

#define HOST_MEM NumaDataFactory<0>
#define DEVICE_MEM NumaDataFactory<1>

//i think for a single grace hopper i only need to use the first 2

constexpr int BLOCK_SIZE = 256;

SpinLock affinity_mutex;


//cuda unified memory
struct ManagedMemoryDataFactory {
    static constexpr bool is_gpu = true; //gpu related memory
    static constexpr int gpu_id = 0;
    uint8_t *data = nullptr; //pointer to allocated memory

    ManagedMemoryDataFactory(size_t size) { //constructor
        cudaMallocManaged(&data, size); //specify managed memory of specific size, when cpu aceesses gpu memory page fault, so we have 2 distinct page tables
    }

    ~ManagedMemoryDataFactory() {
        cudaFree(data);
    }
};

void initialize_memory_pointer_chase(uint8_t *data, size_t size) {
    size_t num_pages = CEIL(size, PAGE_SIZE); //how many pages we cover in size bytes

    size_t *page_sequence = (size_t *) malloc(sizeof(size_t) * num_pages); 
    page_sequence[0] = 0; // first page is the first page, the very first page in the sequence that will be chased (followed by pointers) is fixed to the 0th page of the memory region — that is, the first physical page starting at data, So the pointer chase will start at the beginning of the memory block (data + 0 * PAGE_SIZE), ensuring a known entry point.


    _random_init(time(nullptr), num_pages - 1); //initialize random number generator for randomize the sequence of pages

    for (size_t i = 1; i < num_pages; ++i) {
        page_sequence[i] = _random() + 1; //random sequence of page order we will follow(page sequence is the sequence of pages we will walk)
    }

    for (size_t i = 0; i < num_pages; ++i) { //loop over every page, data is a pointer to the start of the big contiguous memory buffer of our pages
        size_t page_offset = PAGE_SIZE * page_sequence[i]; 
        uint8_t *page_base = data + page_offset; //page_base points to the start of the page_sequence[i]-th page in data
        uint8_t *itr = page_base;
        size_t num_cachelines = std::min(size - page_offset, PAGE_SIZE) / (CACHELINE_SIZE*2); //divide with 2 to space out pointers and avoid cache line conflicts & false sharing, if min!=page size, we measure how many cache lines*2 fit into the currect page(we use min)
        _random_init(time(nullptr), num_cachelines - 1); //random number generator for generating a sequence of cacheline indices inside the page
        for (size_t j = 0; j < num_cachelines - 1; ++j) { //loop over cache line indices INSIDE THE CURRENT PAGE
            uint8_t *new_addr = page_base + (_random() + 1) * (CACHELINE_SIZE*2); //new_addr = address to the next randomized cache line at pointer chase
            *((uint8_t **) itr) = new_addr; //itr:points to a mem location, uint8_t** pointer to a pointer(itr points to a memory location of a pointer),*():means write into that memory location, memory at itr->store the uint8_t* pointer, write at that pointer (*() = new_addr) the new_addr
            itr = new_addr; //Updates itr to point to the new location
        }
        if (i < num_pages - 1) { //if not last page, link the last itr to the address of the beggining of the next page
            // set the last cacheline to point to the next page in the sequence
            *((uint8_t **) itr) = data + PAGE_SIZE * page_sequence[i + 1];
        } else {
            *((uint8_t **) itr) = data; //if last page, circle back to the beggining
        }

    }

    free(page_sequence);

    // invalidate_all(data, size);

    // for (size_t i = 0; i < size; i += (2*CACHELINE_SIZE)) {
    //     *((uint8_t **) &data[i]) = &data[i - (2*CACHELINE_SIZE)];
    // }
}

void initialize_memory_page_chase(uint8_t *data, size_t size) {
	std::cout << "started initialize memory page chase inside memory hpp" << std::endl;
    	size_t num_pages = CEIL(size, PAGE_SIZE);

    _random_init(time(nullptr), num_pages - 1);
    uint8_t *itr = data;
    for (size_t j = 0; j < num_pages - 1; ++j) {
	        std::cout << "num page:" << j << std::endl;

        uint8_t *new_addr = data + (_random() + 1) * PAGE_SIZE;
        *((uint8_t **) itr) = new_addr;
        itr = new_addr;
    }

    *((uint8_t **) itr) = data;

    invalidate_all(data, size); //something like rm the memory region that starts from data and is of size
        std::cout << "done with invalitade" << std::endl;

}

struct MallocDataFactory {
    static constexpr bool is_gpu = false;
    uint8_t *data = nullptr;

    MallocDataFactory(size_t size) {
        data = (uint8_t *) aligned_alloc(64, size); //aligned_alloc(64, size) to allocate size bytes aligned to 64-byte boundaries,important for performance (cache line alignment)
        assert((size_t) data % 64 == 0);
    }

    ~MallocDataFactory() {
        free(data);
    }
};

struct MmapDataFactory {
    uint8_t *data = nullptr;
    size_t size = 0;

    MmapDataFactory(size_t n_bytes) {
        size = n_bytes;
        data = (uint8_t *) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); //memory mapping
    }

    ~MmapDataFactory() {
        munmap(data, size);
    }
};

template <int NODE>
struct NumaDataFactory { //allocate memory on a specific numa node
    static constexpr bool is_gpu = NODE == 1;//predetermined numa node for the 4 gpus
    static constexpr int gpu_id = (NODE);
    uint8_t *data = nullptr;
    size_t size = 0;

    NumaDataFactory(size_t n_bytes) { //constructor
        data = (uint8_t *) numa_alloc_onnode(n_bytes, NODE); //physically allocated ONLY ON CPU on the specific numa node->reduce remote memory access latency!!
        assert((unsigned long) data % PAGE_SIZE == 0);
        // madvise(data, n_bytes, MADV_HUGEPAGE); //Optional hint to use huge pages to reduce TLB misses and improve performance
        memset(data, 0xff, n_bytes); //initialize memory with ff and not 00
        size = n_bytes;
    }

    ~NumaDataFactory() {
        int nodemask = 0;
        get_mempolicy(&nodemask, NULL, 0, data, MPOL_F_NODE | MPOL_F_ADDR);
        assert(nodemask == NODE);
        numa_free(data, size);
    }
};

struct MmioDataFactory {
    uint8_t *data = nullptr;
    size_t size = 0;
    int fd = 0;

    MmioDataFactory(size_t n_bytes) {
        fd = open("/scratch/lfusco/dummy", O_RDWR | O_DIRECT);
        int res = ftruncate(fd, n_bytes); //resize file to n_bytes
        if (res) {
            exit(res);
        }
        lseek(fd, 0, SEEK_SET); //reset file offset
        size = n_bytes;
        data = (uint8_t *) mmap(NULL, n_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); //map file into processes virtual space, so i can use the file like a pointer instead of read(), write()
    }

    ~MmioDataFactory() {
        munmap(data, size);
        close(fd);
    }
};

struct CudaMallocDataFactory {
    static constexpr bool is_gpu = true;
    static constexpr int gpu_id = 0;
    uint8_t *data = nullptr;

    CudaMallocDataFactory(size_t size) {
        cudaMalloc(&data, size);
        cudaMemset(data, 0xff, size); //initialized the data above
    }

    ~CudaMallocDataFactory() {
        cudaFree(data);
    }
};

struct RemoteCudaMallocDataFactory {
    static constexpr bool is_gpu = true;
    static constexpr int gpu_id = 1;
    uint8_t *data = nullptr;

    RemoteCudaMallocDataFactory(size_t size) {
        cudaSetDevice(1);
        cudaMalloc(&data, size);
        cudaMemset(data, 0xff, size);
        cudaSetDevice(0);
    }

    ~RemoteCudaMallocDataFactory() {
        cudaSetDevice(1);
        cudaFree(data);
        cudaSetDevice(0);
    }
};

struct CudaMallocHostDataFactory {
    static constexpr bool is_gpu = false;
    uint8_t *data = nullptr;

    CudaMallocHostDataFactory(size_t size) {
        cudaMallocHost(&data, size);
    }

    ~CudaMallocHostDataFactory() {
        // int nodemask = 0;
        // get_mempolicy(&nodemask, NULL, 0, data, MPOL_F_NODE | MPOL_F_ADDR);
        // printf("NODE %d\n", nodemask);
        cudaFreeHost(data);
    }
};
