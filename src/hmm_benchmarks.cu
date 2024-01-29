#include "base_kernels.cuh"
#include "thread_tools.hpp"

#define CEIL(a, b) (((a)+(b)-1)/(b))

namespace hmm {
    double test_benchmark(size_t n_bytes) {
        size_t size = n_bytes / sizeof(double);
        int grid_size = CEIL(size, BLOCK_SIZE);
        int block_size = size < BLOCK_SIZE ? size : BLOCK_SIZE;
        MallocDataFactory<> data_factory(n_bytes);
        void *args[] = {(void*) &data_factory.data};
        double time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, block_size, args, 0, 0);

        return time;
    }

    double contiguous_host_modified_device_write_benchmark(size_t n_bytes) {
        size_t size = n_bytes / sizeof(double);
        int grid_size = CEIL(size, BLOCK_SIZE);
        int block_size = size < BLOCK_SIZE ? size : BLOCK_SIZE;
        MallocDataFactory<HostModifiedInit<>> data_factory(n_bytes);
        void *args[] = {(void*) &data_factory.data};
        double time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, block_size, args, 0, 0);

        return time;
    }

    double contiguous_device_modified_device_write_benchmark(size_t n_bytes) {
        size_t size = n_bytes / sizeof(double);
        int grid_size = CEIL(size, BLOCK_SIZE);
        int block_size = size < BLOCK_SIZE ? size : BLOCK_SIZE;
        MallocDataFactory<DeviceModifiedInit> data_factory(n_bytes);
        void *args[] = {(void*) &data_factory.data};
        double time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, block_size, args, 0, 0);

        return time;
    }

    double contiguous_untouched_device_write_benchmark(size_t n_bytes) {
        size_t size = n_bytes / sizeof(double);
        int grid_size = CEIL(size, BLOCK_SIZE);
        int block_size = size < BLOCK_SIZE ? size : BLOCK_SIZE;
        MallocDataFactory<> data_factory(n_bytes);
        void *args[] = {(void*) &data_factory.data};
        double time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, block_size, args, 0, 0);

        return time;
    }

    double contiguous_host_invalidated_device_write_benchmark(size_t n_bytes) {
        size_t size = n_bytes / sizeof(double);
        int grid_size = CEIL(size, BLOCK_SIZE);
        int block_size = size < BLOCK_SIZE ? size : BLOCK_SIZE;
        MallocDataFactory<CacheFlushInit> data_factory(n_bytes);
        void *args[] = {(void*) &data_factory.data};
        double time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, block_size, args, 0, 0);

        return time;
    }

    double contiguous_host_exclusive_device_write_benchmark(size_t n_bytes) {
        size_t size = n_bytes / sizeof(double);
        int grid_size = CEIL(size, BLOCK_SIZE);
        int block_size = size < BLOCK_SIZE ? size : BLOCK_SIZE;
        MallocDataFactory<HostModifiedInit<>> data_factory(n_bytes);
        DeviceModifiedInit other_init(data_factory.data, n_bytes);
        void *args[] = {(void*) &data_factory.data};
        double time = time_kernel_execution((void *) strided_write_kernel<double, 1>, grid_size, block_size, args, 0, 0);

        return time;
    }

    double contiguous_untouched_device_write_single_thread_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        dispatch_command(0, INVALIDATE, data_factory.data, n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 1, args, 2, 1, 0, 0);

        return time;
    }

    double contiguous_host_modified_device_write_single_thread_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        dispatch_command(0, WRITE, data_factory.data, n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 1, args, 2, 1, 0, 0);

        return time;
    }

    double contiguous_device_modified_device_write_single_thread_benchmark(size_t n_bytes) {
        MallocDataFactory<DeviceModifiedInit> data_factory(n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 1, args, 2, 1, 0, 0);

        return time;
    }

    double contiguous_untouched_device_write_single_warp_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 32, args, 2, 32, 0, 0);

        return time;
    }

    double contiguous_host_modified_device_write_single_warp_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        dispatch_command(0, WRITE, data_factory.data, n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 32, args, 2, 32, 0, 0);

        return time;
    }

    double contiguous_device_modified_device_write_single_warp_benchmark(size_t n_bytes) {
        MallocDataFactory<DeviceModifiedInit> data_factory(n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 32, args, 2, 32, 0, 0);

        return time;
    }

    double contiguous_untouched_device_write_single_block_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 1024, args, 2, 1024, 0, 0);

        return time;
    }

    double contiguous_host_modified_device_write_single_block_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        dispatch_command(0, WRITE, data_factory.data, n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 1024, args, 2, 1024, 0, 0);

        return time;
    }

    double contiguous_device_modified_device_write_single_block_benchmark(size_t n_bytes) {
        MallocDataFactory<DeviceModifiedInit> data_factory(n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 1, 1024, args, 2, 1024, 0, 0);

        return time;
    }

    double contiguous_untouched_device_write_full_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 264, 1024, args, 2, 264 * 1024, 0, 0);

        return time;
    }

    double contiguous_host_modified_device_write_full_benchmark(size_t n_bytes) {
        MallocDataFactory<> data_factory(n_bytes);
        dispatch_command(0, WRITE, data_factory.data, n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 264, 1024, args, 2, 264 * 1024, 0, 0);

        return time;
    }

    double contiguous_device_modified_device_write_full_benchmark(size_t n_bytes) {
        MallocDataFactory<DeviceModifiedInit> data_factory(n_bytes);
        void *data = data_factory.data;
        void *args[] = {&data, &n_bytes};
        double time = time_kernel_execution_clock((void *) loopy_write_kernel_clock, 264, 1024, args, 2, 264 * 1024, 0, 0);

        return time;
    }
}
