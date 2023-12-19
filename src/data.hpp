#include <immintrin.h>

constexpr int BLOCK_SIZE = 256;

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
        for (size_t i = 0; i < size / 64; i += 64) {
            _mm_clflush(((char *) data) + i);
        }
    }
};

struct HostExclusiveInit : public CacheFlushInit {
    uint64_t dummy = 0;

    HostExclusiveInit(void *data, size_t size) : CacheFlushInit(data, size) {
        for (size_t i = 0; i < size / sizeof(uint64_t); ++i) {
            dummy += ((uint64_t *) data)[i];
        }
    }
};


struct HostModifiedInit : public CacheFlushInit {
    HostModifiedInit(void *data, size_t size) : CacheFlushInit(data, size) {
        memset(data, 0, size);
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
        data = aligned_alloc(64, size);
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
