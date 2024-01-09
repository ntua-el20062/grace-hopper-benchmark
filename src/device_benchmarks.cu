#include "constants.hpp"
#include "base_kernels.cuh"
#include "measurement.hpp"
#include "data.hpp"
#include <string.h>
#include <cstdlib>
#include <stddef.h>

#define CEIL(a, b) (((a)+(b)-1)/(b))

// all sizes should be a power of two!

float contiguous_host_modified_device_read_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostModifiedInit<>> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_read_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_modified_device_read_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceModifiedInit> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_read_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_untouched_device_read_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<NoopInit> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_read_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_cuda_malloc_untouched_device_read_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    CudaMallocDataFactory<NoopInit> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_read_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_cuda_malloc_untouched_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = CEIL(size, BLOCK_SIZE);
    int block_size = size < BLOCK_SIZE ? size : BLOCK_SIZE;
    CudaMallocDataFactory<NoopInit> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, block_size, args, 0, 0);

    return time;
}

float contiguous_cuda_malloc_device_modified_device_read_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    CudaMallocDataFactory<DeviceModifiedInit> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_read_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_host_initialized_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostModifiedInit<>> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_initialized_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceModifiedInit> data_factory(size * sizeof(double));
    strided_write_kernel<double, 1><<<grid_size, BLOCK_SIZE>>>((double *) data_factory.data);
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

// ------------------------------------

float contiguous_untouched_device_copy_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    CudaMallocDataFactory<CacheFlushInit> data_factories[] = {size * sizeof(double), size * sizeof(double)};
    void *args[] = {(void*) data_factories[0].data, (void*) data_factories[1].data};
    float time = time_kernel_execution((void *) strided_copy_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_host_modified_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostModifiedInit<>> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_modified_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceModifiedInit> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_untouched_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<PointerChaseInit<4096>> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float ping_pong_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<NoopInit> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}
