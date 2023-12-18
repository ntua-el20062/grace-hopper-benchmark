#include "constants.hpp"
#include "base_functions.hpp"
#include "base_kernels.cu"
#include "measurement.hpp"
#include "data.hpp"
#include <string.h>
#include <cstdlib>
#include <stddef.h>

#define RUN_BENCHMARK(FUNCNAME, OUTNAME, NITER, BYTES) {\
    float measurements[NITER];\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        measurements[__i] = ((float) BYTES / 1000.f * 1000.f) / FUNCNAME(BYTES);\
    }\
    std::ofstream file(OUTNAME);\
    for (size_t __i = 0; __i < NITER; ++__i) {\
        file << measurements[__i] << std::endl;\
    }}

/**
 * every benchmark is composed of:
 *  - data allocation
 *  - data preparation
 *  - benchmark run
 *  - data free
 * every benchmark returns as a float the execution time
 */

// all sizes should be a power of two!

float contiguous_untouched_device_read_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<NoopDataInitializer> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_read_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_untouched_host_write_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<NoopDataInitializer> data_factory(size * sizeof(double));
    float time = time_function_execution(strided_write_function<double, 1>, (double *) data_factory.data, size);

    return time;
}

float contiguous_host_modified_host_write_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<HostModifiedDataInitializer> data_factory(size * sizeof(double));
    float time = time_function_execution(strided_write_function<double, 1>, (double *) data_factory.data, size);

    return time;
}

float contiguous_untouched_host_copy_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<CacheFlushDataInitializer> out_data_factory(size * sizeof(double));
    MallocDataFactory<CacheFlushDataInitializer> in_data_factory(size * sizeof(double));
    float time = time_function_execution(strided_copy_function<double, 1>, (double *) out_data_factory.data, (double *) in_data_factory.data, size);

    return time;
}

float contiguous_host_exclusive_host_copy_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<HostModifiedDataInitializer> out_data_factory(size * sizeof(double));
    MallocDataFactory<HostModifiedDataInitializer> in_data_factory(size * sizeof(double));
    float time = time_function_execution(strided_copy_function<double, 1>, (double *) out_data_factory.data, (double *) in_data_factory.data, size);

    return time;
}

float contiguous_host_initialized_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostModifiedDataInitializer> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_initialized_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceModifiedDataInitializer> data_factory(size * sizeof(double));
    strided_write_kernel<double, 1><<<grid_size, BLOCK_SIZE>>>((double *) data_factory.data);
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

// ------------------------------------

float contiguous_untouched_device_copy_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    CudaMallocDataFactory<CacheFlushDataInitializer> data_factories[] = {size * sizeof(double), size * sizeof(double)};
    void *args[] = {(void*) data_factories[0].data, (void*) data_factories[1].data};
    float time = time_kernel_execution((void *) strided_copy_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_host_initialized_device_copy_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostModifiedDataInitializer> data_factories[] = {size * sizeof(double), size * sizeof(double)};
    void *args[] = {(void*) data_factories[0].data, (void*) data_factories[1].data};
    float time = time_kernel_execution((void *) strided_copy_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_initialized_device_copy_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceModifiedDataInitializer> data_factories[] = {size * sizeof(double), size * sizeof(double)};
    void *args[] = {(void*) data_factories[0].data, (void*) data_factories[1].data};
    float time = time_kernel_execution((void *) strided_copy_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_host_modified_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<HostModifiedDataInitializer> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_device_modified_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<DeviceModifiedDataInitializer> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}

float contiguous_untouched_device_write_benchmark(size_t size) {
    size /= sizeof(double);
    int grid_size = size / BLOCK_SIZE;
    ManagedMemoryDataFactory<NoopDataInitializer> data_factory(size * sizeof(double));
    void *args[] = {(void*) &data_factory.data, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    float time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, BLOCK_SIZE, args, 0, 0);

    return time;
}