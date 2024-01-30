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

struct NoopInit {
    NoopInit(void *data, size_t size) {}
};

template <size_t STRIDE, typename SUBINITIALIZER = NoopInit>
struct PointerChaseInit {
    PointerChaseInit(void *data, size_t size) {
        static_assert(STRIDE >= sizeof(void *));
        memset(data, 0, size);
        char *bytedata = (char *) data;
        size_t num_hops = size / STRIDE;
        if (num_hops == 0) {
            *((void **) data) = nullptr;
        } else {
            for (size_t i = 0; i < num_hops - 1; ++i) {
                *((void **)(bytedata + i * STRIDE)) = (void *) (bytedata + (i + 1) * STRIDE);
            }
            *((void **)(bytedata + (num_hops - 1) * STRIDE)) = nullptr;
        }
        SUBINITIALIZER subinitializer(data, size);
    }
};

struct CacheFlushInit {
    CacheFlushInit(void *data, size_t size) {
        for (size_t i = 0; i < size; i += 64) {
            char *cur_line = ((char *) data) + i;
            asm volatile("dc civac, %0" : : "r"(cur_line) : "memory");
        }
        asm volatile("" ::: "memory");
    }
};


template <int TID = 0>
struct HostExclusiveInit : public CacheFlushInit {
    uint64_t dummy = 0;

    static void inner_function(void *data, size_t size, uint64_t &dummy) {
        affinity_mutex.lock();
        CacheFlushInit init(data, size);
        for (size_t i = 0; i < size / sizeof(uint64_t); ++i) {
            dummy += ((uint64_t *) data)[i];
        }

        printf("%lu", dummy);
        asm volatile("" ::: "memory");
        affinity_mutex.unlock();
    }

    HostExclusiveInit(void *data, size_t size) : CacheFlushInit(data, size) {
        if constexpr (TID == 0) {
            for (size_t i = 0; i < size / sizeof(uint64_t); ++i) {
                dummy += ((uint64_t *) data)[i];
            }

            printf("%lu", dummy);
        } else {
            affinity_mutex.lock();
            std::thread t(HostExclusiveInit::inner_function, data, size, std::ref(dummy));
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(1, &cpuset);  // set affinity to CPU 1

            pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
            affinity_mutex.unlock();

            t.join();
        }
        asm volatile("" ::: "memory");
    }
};

template <int TID = 0>
struct HostModifiedInit : public CacheFlushInit {
    static void inner_function(void *data, size_t size) {
        affinity_mutex.lock();
        CacheFlushInit init(data, size);
        memset(data, 0xff, size);
        asm volatile("" ::: "memory");
        affinity_mutex.unlock();
    }

    HostModifiedInit(void *data, size_t size) : CacheFlushInit(data, size) {
        if constexpr (TID == 0) {
            memset(data, 0xff, size);
        } else {
            affinity_mutex.lock();
            std::thread t(HostModifiedInit::inner_function, data, size);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(1, &cpuset);  // set affinity to CPU 1

            pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
            affinity_mutex.unlock();

            t.join();
        }
        asm volatile("" ::: "memory");
    }
};

struct DeviceExclusiveInit {
    DeviceExclusiveInit(void *data, size_t size) {
        strided_read_kernel<double, 1><<<(size / sizeof(double)) / BLOCK_SIZE, BLOCK_SIZE>>>((double *) data);
        cudaDeviceSynchronize();
    }
};

struct DeviceModifiedInit {
    DeviceModifiedInit(void *data, size_t size) {
        device_modified_init_kernel<<<1, 1>>>((uint8_t *) data, size);
        cudaDeviceSynchronize();
    }
};

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

    size_t page_sequence[num_pages];
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
        }
    }

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

    invalidate_all(data, size);
}

template <typename DATA_INITIALIZER = NoopInit>
struct MallocDataFactory {
    uint8_t *data = nullptr;

    MallocDataFactory(size_t size) {
        data = (uint8_t *) aligned_alloc(64, size);
        assert((size_t) data % 64 == 0);
        DATA_INITIALIZER initializer(data, size);
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

template <typename DATA_INITIALIZER = NoopInit>
struct CudaMallocDataFactory {
    uint8_t *data = nullptr;

    CudaMallocDataFactory(size_t size) {
        cudaMalloc(&data, size);
        DATA_INITIALIZER initializer(data, size);
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
