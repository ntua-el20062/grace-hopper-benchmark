#include <immintrin.h>

constexpr int BLOCK_SIZE = 256;

struct NoopDataInitializer {
    NoopDataInitializer(void *data, size_t size) {}
};

struct CacheFlushDataInitializer {
    CacheFlushDataInitializer(void *data, size_t size) {
        for (size_t i = 0; i < size / 64; i += 64) {
            _mm_clflush(((char *) data) + i);
        }
    }
};

struct HostExclusiveDataInitializer : public CacheFlushDataInitializer {
    uint64_t dummy = 0;

    HostExclusiveDataInitializer(void *data, size_t size) : CacheFlushDataInitializer(data, size) {
        for (size_t i = 0; i < size / sizeof(uint64_t); ++i) {
            dummy += ((uint64_t *) data)[i];
        }
    }
};


struct HostModifiedDataInitializer : public CacheFlushDataInitializer {
    HostModifiedDataInitializer(void *data, size_t size) : CacheFlushDataInitializer(data, size) {
        memset(data, 0, size);
    }
};

struct DeviceExclusiveDataInitializer {
    DeviceExclusiveDataInitializer(void *data, size_t size) {
        strided_read_kernel<double, 1><<<(size / sizeof(double)) / BLOCK_SIZE, BLOCK_SIZE>>>((double *) data);
        cudaDeviceSynchronize();
    }
};

struct DeviceModifiedDataInitializer {
    DeviceModifiedDataInitializer(void *data, size_t size) {
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
    void *data;

    MallocDataFactory(size_t size) {
        data = aligned_alloc(64, size);
        DATA_INITIALIZER initializer(data, size);
    }

    ~MallocDataFactory() {
        free(data);
    }
};

template <typename DATA_INITIALIZER>
struct CudaMallocDataFactory {
    void *data = (void *) 0xdeadbeefdeadbeef;

    CudaMallocDataFactory(size_t size) {
        cudaMalloc(&data, size);
        DATA_INITIALIZER initializer(data, size);
    }

    ~CudaMallocDataFactory() {
        cudaFree(data);
    }
};
