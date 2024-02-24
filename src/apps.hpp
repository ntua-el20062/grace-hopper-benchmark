#pragma once

#include "measurement.hpp"
#include "data.hpp"
#include "thread_tools.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <time.h>
// #include <cudnn.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#ifdef OPENBLAS
#include <cblas.h>
#include <openblas_config.h>
#endif

cublasHandle_t cublas;
// cudnnHandle_t cudnn;

#define CHECK_CUBLAS(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << #call << ": " << cublasGetStatusName(status) << ": " << cublasGetStatusString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void init_cublas() {
    CHECK_CUBLAS(cublasCreate(&cublas);)
}

// void init_cudnn() {
//     cudnnCreate(&cudnn);
// }

template <typename T>
void fill_random(T *arr, size_t n_bytes) {
    size_t n_elems = n_bytes / sizeof(T);
    for (size_t i = 0; i < n_elems; ++i) {
        arr[i] = (T) rand() / (T) RAND_MAX;
    }
}

template <typename A_ALLOC, typename B_ALLOC, typename C_ALLOC>
void cublas_gemm_template(size_t n_iter, size_t n_bytes, size_t a_target, size_t b_target, size_t c_target, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(float);
    size_t matrix_side = std::sqrt(n_elems);

    A_ALLOC A(n_elems * sizeof(float));
    B_ALLOC B(n_elems * sizeof(float));
    C_ALLOC C(n_elems * sizeof(float));
    dispatch_command(a_target, WRITE, A.data, n_elems * sizeof(float));
    dispatch_command(b_target, WRITE, B.data, n_elems * sizeof(float));
    dispatch_command(c_target, WRITE, C.data, n_elems * sizeof(float));

    fill_random((float *) A.data, n_elems * sizeof(float));
    fill_random((float *) B.data, n_elems * sizeof(float));
    // memset(C.data, 0, n_elems * sizeof(float));

    float alpha = 1.f;
    float beta = 0.f;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        uint64_t start = get_cpu_clock();
        CHECK_CUBLAS(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                matrix_side, matrix_side, matrix_side,
                &alpha,
                (float *) A.data, matrix_side,
                (float *) B.data, matrix_side,
                &beta,
                (float *) C.data, matrix_side);)
        cudaDeviceSynchronize();
        uint64_t end = get_cpu_clock();
        times[iter] = get_elapsed_milliseconds_clock(start, end);
    }

    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/apps/gemm/cublas/" + name);
}

#ifdef OPENBLAS
template <typename A_ALLOC, typename B_ALLOC, typename C_ALLOC>
void openblas_gemm_template(size_t n_iter, size_t n_bytes, size_t a_target, size_t b_target, size_t c_target, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(float);
    size_t matrix_side = std::sqrt(n_elems);

    float alpha = 1.0;
    float beta = 0.0;

    A_ALLOC A(n_elems * sizeof(float));
    B_ALLOC B(n_elems * sizeof(float));
    C_ALLOC C(n_elems * sizeof(float));
    dispatch_command(a_target, WRITE, A.data, n_elems * sizeof(float));
    dispatch_command(b_target, WRITE, B.data, n_elems * sizeof(float));
    dispatch_command(c_target, WRITE, C.data, n_elems * sizeof(float));

    fill_random((float *) A.data, n_elems * sizeof(float));
    fill_random((float *) B.data, n_elems * sizeof(float));
    
    openblas_set_num_threads(72);

    for (size_t iter = 0; iter < n_iter; ++iter) {
        uint64_t start = get_cpu_clock();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                matrix_side, matrix_side, matrix_side, alpha,
                (float *) A.data, matrix_side,
                (float *) B.data, matrix_side, beta,
                (float *) C.data, matrix_side);
        uint64_t end = get_cpu_clock();
        times[iter] = get_elapsed_milliseconds_clock(start, end);
    }

    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/apps/gemm/openblas/" + name);
}

void run_openblas_gemm_tests(size_t n_iter, size_t n_bytes) {
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_ddr_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_ddr_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_hbm_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, DEVICE_ID, "ddr_hbm_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_hbm_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, REMOTE_DEVICE_ID, "ddr_hbm_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_ddr_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_hbm_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, DEVICE_ID, "ddr_hbm_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "ddr_hbm_remote_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_ddr_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_ddr_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_hbm_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, DEVICE_ID, "hbm_hbm_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_hbm_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, REMOTE_DEVICE_ID, "hbm_hbm_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_ddr_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_hbm_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, DEVICE_ID, "hbm_hbm_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "hbm_hbm_remote_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_remote_ddr_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_remote_ddr_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_remote_hbm_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, DEVICE_ID, "ddr_remote_hbm_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_remote_hbm_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, REMOTE_DEVICE_ID, "ddr_remote_hbm_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_remote_ddr_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_remote_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_remote_hbm_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, DEVICE_ID, "ddr_remote_hbm_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_remote_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "ddr_remote_hbm_remote_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_remote_ddr_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_remote_ddr_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_remote_hbm_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, DEVICE_ID, "hbm_remote_hbm_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_remote_hbm_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, REMOTE_DEVICE_ID, "hbm_remote_hbm_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_remote_ddr_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_remote_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_remote_hbm_remote_ddr/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, DEVICE_ID, "hbm_remote_hbm_remote_hbm/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_remote_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    openblas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "hbm_remote_hbm_remote_hbm_remote/" + std::to_string(n_bytes));

}
#endif

void run_cublas_gemm_tests(size_t n_iter, size_t n_bytes) {
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_ddr_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_ddr_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_hbm_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, DEVICE_ID, "ddr_hbm_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_hbm_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, REMOTE_DEVICE_ID, "ddr_hbm_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_ddr_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_hbm_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, DEVICE_ID, "ddr_hbm_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "ddr_hbm_remote_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_ddr_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_ddr_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_hbm_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, DEVICE_ID, "hbm_hbm_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_hbm_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, REMOTE_DEVICE_ID, "hbm_hbm_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_ddr_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_hbm_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, DEVICE_ID, "hbm_hbm_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, DEVICE_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "hbm_hbm_remote_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_remote_ddr_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_remote_ddr_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_remote_hbm_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, DEVICE_ID, "ddr_remote_hbm_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, DEVICE_ID, HOST_ID, "ddr_remote_hbm_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, DEVICE_ID, REMOTE_DEVICE_ID, "ddr_remote_hbm_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_remote_ddr_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr_remote_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, HOST_ID, HOST_ID, REMOTE_DEVICE_ID, "ddr_remote_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_remote_hbm_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, DEVICE_ID, "ddr_remote_hbm_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, HOST_ID, "ddr_remote_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<NumaDataFactory<1>, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, HOST_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "ddr_remote_hbm_remote_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_remote_ddr_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_remote_ddr_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_remote_hbm_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, DEVICE_ID, "hbm_remote_hbm_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_remote_hbm_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, DEVICE_ID, REMOTE_DEVICE_ID, "hbm_remote_hbm_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_remote_ddr_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, HOST_ID, "hbm_remote_ddr_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, NumaDataFactory<1>, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, HOST_ID, REMOTE_DEVICE_ID, "hbm_remote_ddr_remote_hbm_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_remote_hbm_remote_ddr/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, DEVICE_ID, "hbm_remote_hbm_remote_hbm/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, NumaDataFactory<1>>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, HOST_ID, "hbm_remote_hbm_remote_ddr_remote/" + std::to_string(n_bytes));
    cublas_gemm_template<MmapDataFactory, MmapDataFactory, MmapDataFactory>(n_iter, n_bytes, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, REMOTE_DEVICE_ID, "hbm_remote_hbm_remote_hbm_remote/" + std::to_string(n_bytes));
}


// void cudnn_convolution_template(size_t n_iter, size_t n_bytes, size_t a_target, size_t b_target, size_t c_target, std::string name) {
//     double times[n_iter];
//     size_t n_elems = n_bytes / sizeof(float);
//     size_t matrix_side = std::sqrt(n_elems);

//     // Define input dimensions
//     int batch_size = 1, channels = 1, height = matrix_side, width = matrix_side;
//     cudnnTensorDescriptor_t input_descriptor;
//     cudnnCreateTensorDescriptor(&input_descriptor);
//     cudnnSetTensor4dDescriptor(input_descriptor,
//                                CUDNN_TENSOR_NCHW,
//                                CUDNN_DATA_FLOAT,
//                                batch_size,
//                                channels,
//                                height,
//                                width);

//     // Define kernel (filter) dimensions
//     int out_channels = 1, kernel_height = 3, kernel_width = 3;
//     cudnnFilterDescriptor_t kernel_descriptor;
//     cudnnCreateFilterDescriptor(&kernel_descriptor);
//     cudnnSetFilter4dDescriptor(kernel_descriptor,
//                                CUDNN_DATA_FLOAT,
//                                CUDNN_TENSOR_NCHW,
//                                out_channels,
//                                channels,
//                                kernel_height,
//                                kernel_width);

//     // Define convolution parameters
//     cudnnConvolutionDescriptor_t convolution_descriptor;
//     cudnnCreateConvolutionDescriptor(&convolution_descriptor);
//     cudnnSetConvolution2dDescriptor(convolution_descriptor,
//                                     0, 0, // padding
//                                     1, 1, // stride
//                                     1, 1, // dilation
//                                     CUDNN_CROSS_CORRELATION,
//                                     CUDNN_DATA_FLOAT);

//     // Calculate output dimensions and allocate memory
//     int n, c, h, w;
//     cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
//                                           input_descriptor,
//                                           kernel_descriptor,
//                                           &n, &c, &h, &w);

//     cudnnTensorDescriptor_t output_descriptor;
//     cudnnCreateTensorDescriptor(&output_descriptor);
//     cudnnSetTensor4dDescriptor(output_descriptor,
//                                CUDNN_TENSOR_NCHW,
//                                CUDNN_DATA_FLOAT,
//                                n, c, h, w);

//     // Allocate memory for input, kernel, and output on device
//     MmapDataFactory input(batch_size * channels * height * width * sizeof(float));
//     MmapDataFactory kernel(out_channels * channels * kernel_height * kernel_width * sizeof(float));
//     MmapDataFactory output(n * c * h * w * sizeof(float));
//     dispatch_command(a_target, WRITE, A.data, n_elems * sizeof(float));
//     dispatch_command(b_target, WRITE, B.data, n_elems * sizeof(float));
//     dispatch_command(c_target, WRITE, C.data, n_elems * sizeof(float));

//     // Perform convolution
//     float alpha = 1.0f, beta = 0.0f;
//     cudnnConvolutionFwdAlgo_t convolution_algorithm;
//     cudnnGetConvolutionForwardAlgorithm(cudnn,
//                                         input_descriptor,
//                                         kernel_descriptor,
//                                         convolution_descriptor,
//                                         output_descriptor,
//                                         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//                                         0,
//                                         &convolution_algorithm);

//     size_t workspace_bytes = 0;
//     cudnnGetConvolutionForwardWorkspaceSize(cudnn,
//                                             input_descriptor,
//                                             kernel_descriptor,
//                                             convolution_descriptor,
//                                             output_descriptor,
//                                             convolution_algorithm,
//                                             &workspace_bytes);

//     void* d_workspace{nullptr};
//     cudaMalloc(&d_workspace, workspace_bytes);
//     std::cout << "workspace bytes: " << workspace_bytes << std::endl;

//     for (size_t iter = 0; iter < n_iter; ++iter) {
//         uint64_t start = get_cpu_clock();
//         cudnnConvolutionForward(cudnn,
//                                 &alpha,
//                                 input_descriptor,
//                                 (float *) input.data,
//                                 kernel_descriptor,
//                                 (float *) kernel.data,
//                                 convolution_descriptor,
//                                 convolution_algorithm,
//                                 d_workspace,
//                                 workspace_bytes,
//                                 &beta,
//                                 output_descriptor,
//                                 (float *) output.data);
//         cudaDeviceSynchronize();
//         uint64_t end = get_cpu_clock();
//         times[iter] = get_elapsed_milliseconds_clock(start, end);
//     }

//     millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/apps/conv/cudnn/" + name);

//     cudaFree(d_workspace);
//     cudnnDestroyTensorDescriptor(input_descriptor);
//     cudnnDestroyTensorDescriptor(output_descriptor);
//     cudnnDestroyFilterDescriptor(kernel_descriptor);
//     cudnnDestroyConvolutionDescriptor(convolution_descriptor);
// }

// void run_cudnn_convolution_tests(size_t n_iter, size_t n_bytes) {
//     cudnn_convolution_template(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr/" + std::to_string(n_bytes)); // ddr uniform
//     cudnn_convolution_template(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, DEVICE_ID, "hbm/" + std::to_string(n_bytes)); // hbm uniform
//     cudnn_convolution_template(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_hbm_ddr/" + std::to_string(n_bytes)); // hbm -> ddr
//     cudnn_convolution_template(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_ddr_hbm/" + std::to_string(n_bytes)); // ddr->hbm
//     cudnn_convolution_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_ddr/" + std::to_string(n_bytes)); // het->ddr
//     cudnn_convolution_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "ddr_hbm_ddr/" + std::to_string(n_bytes)); // het->ddr
//     cudnn_convolution_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_ddr_hbm/" + std::to_string(n_bytes)); // het->hbm
//     cudnn_convolution_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "ddr_hbm_hbm/" + std::to_string(n_bytes)); // het->hbm
// }

template <typename ALLOC, typename Policy>
void thrust_reduction_template(size_t n_iter, size_t n_bytes, const Policy &exec, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(float);

    ALLOC factory(n_elems * sizeof(float));
    float *data_ptr = (float *) factory.data;
    for (size_t i = 0; i < n_elems; ++i) {
        data_ptr[i] = i + 1;
    }
    float expected = (n_elems*(n_elems+1))/2;

    for (size_t iter = 0; iter < n_iter; ++iter) {
        uint64_t start = get_cpu_clock();
        float res = thrust::reduce(exec, data_ptr, data_ptr + n_elems);
        uint64_t end = get_cpu_clock();
        times[iter] = get_elapsed_milliseconds_clock(start, end);
        if(res != expected) {
            printf("%f vs expected %f, %s\n", res, expected, name.c_str());
        }
    }

    millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/apps/reduce/" + name);
}

void run_thrust_reduction_benchmarks(size_t n_iter, size_t n_bytes) {
    thrust_reduction_template<HOST_MEM>(n_iter, n_bytes, thrust::host, "host/ddr/" + std::to_string(n_bytes));
    thrust_reduction_template<DEVICE_MEM>(n_iter, n_bytes, thrust::host, "host/hbm/" + std::to_string(n_bytes));
    thrust_reduction_template<REMOTE_HOST_MEM>(n_iter, n_bytes, thrust::host, "host/ddr_remote/" + std::to_string(n_bytes));
    thrust_reduction_template<REMOTE_DEVICE_MEM>(n_iter, n_bytes, thrust::host, "host/hbm_remote/" + std::to_string(n_bytes));
    thrust_reduction_template<HOST_MEM>(n_iter, n_bytes, thrust::device, "device/ddr/" + std::to_string(n_bytes));
    thrust_reduction_template<DEVICE_MEM>(n_iter, n_bytes, thrust::device, "device/hbm/" + std::to_string(n_bytes));
    thrust_reduction_template<REMOTE_HOST_MEM>(n_iter, n_bytes, thrust::device, "device/ddr_remote/" + std::to_string(n_bytes));
    thrust_reduction_template<REMOTE_DEVICE_MEM>(n_iter, n_bytes, thrust::device, "device/hbm_remote/" + std::to_string(n_bytes));
}