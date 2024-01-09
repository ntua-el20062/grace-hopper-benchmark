#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <mutex>
#include <thread>
#include "thread_tools.hpp"

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
        // memset(data, 0xff, size);
        for (size_t i = 0; i < size; i += 64) {
            char *cur_line = ((char *) data) + i;
#ifdef __x86_64__
            _mm_clflush(cur_line);
#else
            asm volatile("dc civac, %0" : : "r"(cur_line) : "memory");
#endif
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
        strided_write_kernel<double, 1><<<(size / sizeof(double)) / BLOCK_SIZE, BLOCK_SIZE>>>((double *) data);
        cudaDeviceSynchronize();
    }
};

template <typename DATA_INITIALIZER>
struct ManagedMemoryDataFactory {
    void *data = nullptr;

    ManagedMemoryDataFactory(size_t size) {
        cudaMallocManaged(&data, size);
        DATA_INITIALIZER initializer(data, size);
    }

    ~ManagedMemoryDataFactory() {
        cudaFree(data);
    }
};

template <typename DATA_INITIALIZER>
struct MallocDataFactory {
    void *data = nullptr;

    MallocDataFactory(size_t size) {
        data = aligned_alloc(128, size);
        DATA_INITIALIZER initializer(data, size);
    }

    ~MallocDataFactory() {
        free(data);
    }
};

template <typename DATA_INITIALIZER>
struct CudaMallocDataFactory {
    void *data = nullptr;

    CudaMallocDataFactory(size_t size) {
        cudaMalloc(&data, size);
        DATA_INITIALIZER initializer(data, size);
    }

    ~CudaMallocDataFactory() {
        cudaFree(data);
    }
};
