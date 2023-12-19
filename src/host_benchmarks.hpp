#include "base_functions.hpp"

float contiguous_host_modified_host_write_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<HostModifiedInit> data_factory(size * sizeof(double));
    float time = time_function_execution(strided_write_function<double, 1>, (double *) data_factory.data, size);

    return time;
}

float contiguous_untouched_host_copy_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<CacheFlushInit> out_data_factory(size * sizeof(double));
    MallocDataFactory<CacheFlushInit> in_data_factory(size * sizeof(double));
    float time = time_function_execution(strided_copy_function<double, 1>, (double *) out_data_factory.data, (double *) in_data_factory.data, size);

    return time;
}

float contiguous_host_exclusive_host_copy_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<HostModifiedInit> out_data_factory(size * sizeof(double));
    MallocDataFactory<HostModifiedInit> in_data_factory(size * sizeof(double));
    float time = time_function_execution(strided_copy_function<double, 1>, (double *) out_data_factory.data, (double *) in_data_factory.data, size);

    return time;
}

float contiguous_untouched_host_write_benchmark(size_t size) {
    size /= sizeof(double);
    MallocDataFactory<NoopInit> data_factory(size * sizeof(double));
    float time = time_function_execution(strided_write_function<double, 1>, (double *) data_factory.data, size);

    return time;
}