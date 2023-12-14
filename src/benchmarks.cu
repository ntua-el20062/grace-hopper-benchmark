#include "constants.hpp"
#include "base_functions.hpp"
#include "base_kernels.cu"
#include "measurement.hpp"
#include <string.h>
#include <cstdlib>

constexpr int BLOCK_SIZE = 256;

/**
 * every benchmark is composed of:
 *  - data allocation
 *  - data preparation
 *  - benchmark run
 *  - data free
 * every benchmark returns as a float the execution time
 */

// all sizes should be a power of two!

struct VoidDataInitializer {
    VoidDataInitializer(void *data, size_t size) {}
};

struct HostDataInitializer {
    HostDataInitializer(void *data, size_t size) {
        memset(data, 0, size);
    }
};

struct DeviceDataInitializer {
    DeviceDataInitializer(void *data, size_t size) {
        strided_write_kernel<char, 0><<<size / BLOCK_SIZE, BLOCK_SIZE>>>((char *) data);
        cudaDeviceSynchronize();
    }
};

template <typename DATA_INITIALIZER>
struct ManagedMemoryDataFactory {
    void *data;

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
        data = malloc(size);
        DATA_INITIALIZER initializer(data, size);
    }

    ~MallocDataFactory() {
        free(data);
    }
};

template <typename DATA_INITIALIZER>
struct DeviceCudaMallocDataFactory {
    void *data;

    DeviceCudaMallocDataFactory(size_t size) {
        cudaMalloc(&data, size);
        DATA_INITIALIZER initializer(data, size);
    }

    ~DeviceCudaMallocDataFactory() {
        cudaFree(data);
    }
};

float contiguous_untouched_float_device_write_benchmark(size_t size) {
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<VoidDataInitializer> data_factory(size * sizeof(float));
    void *args[] = {(void*) data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<float, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_untouched_float_device_read_benchmark(size_t size) {
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<VoidDataInitializer> data_factory(size * sizeof(float));
    void *args[] = {(void*) data_factory.data};
    float time = time_kernel_execution((void *) strided_read_kernel<float, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_untouched_float_host_write_benchmark(size_t size) {
    ManagedMemoryDataFactory<VoidDataInitializer> data_factory(size * sizeof(float));
    float time = time_function_execution(strided_write_function<float, 1>, (float *) data_factory.data, size);

    return time;
}

float contiguous_host_initialized_float_device_write_benchmark(size_t size) {
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostDataInitializer> data_factory(size * sizeof(float));
    void *args[] = {(void*) data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<float, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_initialized_float_device_write_benchmark(size_t size) {
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceDataInitializer> data_factory(size * sizeof(float));
    strided_write_kernel<float, 1><<<grid_size, BLOCK_SIZE>>>((float *) data_factory.data);
    void *args[] = {(void*) data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<float, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

// ------------------------------------

float contiguous_untouched_float_device_copy_benchmark(size_t size) {
    int grid_size = size / BLOCK_SIZE;
    DeviceCudaMallocDataFactory<VoidDataInitializer> data_factories[] = {size * sizeof(float), size * sizeof(float)};
    void *args[] = {(void*) data_factories[0].data, (void*) data_factories[1].data};
    float time = time_kernel_execution((void *) strided_copy_kernel<float, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_host_initialized_float_device_copy_benchmark(size_t size) {
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostDataInitializer> data_factories[] = {size * sizeof(float), size * sizeof(float)};
    void *args[] = {(void*) data_factories[0].data, (void*) data_factories[1].data};
    float time = time_kernel_execution((void *) strided_copy_kernel<float, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_initialized_float_device_copy_benchmark(size_t size) {
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceDataInitializer> data_factories[] = {size * sizeof(float), size * sizeof(float)};
    void *args[] = {(void*) data_factories[0].data, (void*) data_factories[1].data};
    float time = time_kernel_execution((void *) strided_copy_kernel<float, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}
