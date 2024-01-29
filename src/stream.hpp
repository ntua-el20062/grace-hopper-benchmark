#pragma once

#include "data.hpp"
#include "measurement.hpp"

template <typename SRC_ALLOC, typename DST_ALLOC>
void stream_test_device_template(size_t n_bytes, size_t n_iter, int grid_size, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    double times[n_iter];
    size_t per_array_bytes = n_bytes / 2;
    size_t n_elems = per_array_bytes / sizeof(double);

    cudaDeviceReset();
    DST_ALLOC dst(per_array_bytes);
    SRC_ALLOC src(per_array_bytes);
    dispatch_command(dst_target, dst_mode, dst.data, per_array_bytes);
    dispatch_command(src_target, src_mode, src.data, per_array_bytes);

    void *args[] = {(void *) &dst.data, (void *) &src.data, (void *) &n_elems};
    for (size_t i = 0; i < n_iter; ++i) {
        times[i] = time_kernel_execution((void *) stream_copy<double>, grid_size, 1024, args, 0, 0);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/stream/copy/device/" + name + "/" + std::to_string(n_bytes));
}

void run_stream_benchmark_device(size_t n_bytes, size_t n_iter, int grid_size) {
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, NONE, HOST_ID, NONE, "uninitialized_uninitialized");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, NONE, HOST_ID, READ, "uninitialized_host_read");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, NONE, HOST_ID, WRITE, "uninitialized_host_write");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, READ, HOST_ID, NONE, "host_read_uninitialized");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, WRITE, HOST_ID, NONE, "host_write_uninitialized");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, READ, HOST_ID, READ, "host_read_host_read");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, WRITE, HOST_ID, WRITE, "host_write_host_write");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, READ, HOST_ID, WRITE, "host_read_host_write");
    stream_test_device_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, grid_size, HOST_ID, WRITE, HOST_ID, READ, "host_write_host_read");
}

template <typename T>
__attribute__((always_inline)) inline void host_stream_copy(T *dst, T *src, size_t n_elems) {
    #pragma omp parallel for
    for (size_t i = 0; i < n_elems; ++i) {
        dst[i] = src[i];
    }
}

template <typename DST_ALLOC, typename SRC_ALLOC>
void stream_test_host_template(size_t n_bytes, size_t n_iter, size_t src_target, ThreadCommand src_mode, size_t dst_target, ThreadCommand dst_mode, std::string name) {
    double times[n_iter];
    size_t per_array_bytes = n_bytes / 2;
    size_t n_elems = per_array_bytes / sizeof(double);

    DST_ALLOC dst(per_array_bytes);
    SRC_ALLOC src(per_array_bytes);
    dispatch_command(dst_target, dst_mode, dst.data, per_array_bytes);
    dispatch_command(src_target, src_mode, src.data, per_array_bytes);

    for (size_t i = 0; i < n_iter; ++i) {
        TIME_FUNCTION_EXECUTION(times[i], (host_stream_copy<double>), (double *) dst.data, (double *) src.data, n_elems);
    }
    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/stream/copy/host/" + name + "/" + std::to_string(n_bytes));
}

void run_stream_benchmark_host(size_t n_iter, size_t n_bytes) {
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, HOST_ID, NONE, HOST_ID, NONE, "uninitialized_uninitialized");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, HOST_ID, READ, HOST_ID, READ, "host_read_host_read");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, HOST_ID, NONE, DEVICE_ID, READ, "uninitialized_device_read");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, HOST_ID, NONE, DEVICE_ID, WRITE, "uninitialized_device_write");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, DEVICE_ID, READ, HOST_ID, NONE, "device_read_uninitialized");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, DEVICE_ID, WRITE, HOST_ID, NONE, "device_write_uninitialized");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, DEVICE_ID, READ, DEVICE_ID, READ, "device_read_device_read");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, DEVICE_ID, WRITE, DEVICE_ID, WRITE, "device_write_device_write");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, DEVICE_ID, READ, DEVICE_ID, WRITE, "device_read_device_write");
    stream_test_host_template<MmapDataFactory, MmapDataFactory>(n_bytes, n_iter, DEVICE_ID, WRITE, DEVICE_ID, READ, "device_write_device_read");
}