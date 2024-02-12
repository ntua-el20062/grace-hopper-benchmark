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

constexpr int BLOCK_SIZE = 256;

SpinLock affinity_mutex;

struct ManagedMemoryDataFactory {
    uint8_t *data = nullptr;

    ManagedMemoryDataFactory(size_t size) {
        cudaMallocManaged(&data, size);
    }

    ~ManagedMemoryDataFactory() {
        cudaFree(data);
    }
};

void initialize_memory_pointer_chase(uint8_t *data, size_t size) {
    size_t num_pages = CEIL(size, PAGE_SIZE);

    size_t *page_sequence = (size_t *) malloc(sizeof(size_t) * num_pages);
    page_sequence[0] = 0; // first page is the first page
    _random_init(0, num_pages - 1);

    for (size_t i = 1; i < num_pages; ++i) {
        page_sequence[i] = _random() + 1;
    }

    for (size_t i = 0; i < num_pages; ++i) {
        size_t page_offset = PAGE_SIZE * page_sequence[i];
        uint8_t *page_base = data + page_offset;
        uint8_t *itr = page_base;
        size_t num_cachelines = std::min(size - page_offset, PAGE_SIZE) / CACHELINE_SIZE;
        _random_init(0, num_cachelines - 1);
        for (size_t j = 0; j < num_cachelines - 1; ++j) {
            uint8_t *new_addr = page_base + (_random() + 1) * CACHELINE_SIZE;
            *((uint8_t **) itr) = new_addr;
            itr = new_addr;
        }
        if (i < num_pages - 1) {
            // set the last cacheline to point to the next page in the sequence
            *((uint8_t **) itr) = data + PAGE_SIZE * page_sequence[i + 1];
        } else {
            *((uint8_t **) itr) = data;
        }

    }

    free(page_sequence);

    invalidate_all(data, size);
}

void initialize_memory_page_chase(uint8_t *data, size_t size) {
    size_t num_pages = CEIL(size, PAGE_SIZE);

    _random_init(0, num_pages - 1);
    uint8_t *itr = data;
    for (size_t j = 0; j < num_pages - 1; ++j) {
        uint8_t *new_addr = data + (_random() + 1) * PAGE_SIZE;
        *((uint8_t **) itr) = new_addr;
        itr = new_addr;
    }

    *((uint8_t **) itr) = data;

    invalidate_all(data, size);
}

struct MallocDataFactory {
    uint8_t *data = nullptr;

    MallocDataFactory(size_t size) {
        data = (uint8_t *) aligned_alloc(64, size);
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
        data = (uint8_t *) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        madvise(data, size, MADV_HUGEPAGE);
    }

    ~MmapDataFactory() {
        munmap(data, size);
    }
};

struct MmioDataFactory {
    uint8_t *data = nullptr;
    size_t size = 0;
    int fd = 0;

    MmioDataFactory(size_t n_bytes) {
        fd = open("/scratch/lfusco/dummy", O_RDWR | O_DIRECT);
        int res = ftruncate(fd, n_bytes);
        if (res) {
            exit(res);
        }
        lseek(fd, 0, SEEK_SET);
        size = n_bytes;
        data = (uint8_t *) mmap(NULL, n_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }

    ~MmioDataFactory() {
        munmap(data, size);
        close(fd);
    }
};

struct CudaMallocDataFactory {
    uint8_t *data = nullptr;

    CudaMallocDataFactory(size_t size) {
        cudaMalloc(&data, size);
    }

    ~CudaMallocDataFactory() {
        cudaFree(data);
    }
};

struct CudaMallocHostDataFactory {
    uint8_t *data = nullptr;

    CudaMallocHostDataFactory(size_t size) {
        cudaMallocHost(&data, size);
    }

    ~CudaMallocHostDataFactory() {
        cudaFreeHost(data);
    }
};
