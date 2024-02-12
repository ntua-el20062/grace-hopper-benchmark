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
// #include <arm_compute/runtime/NEON/NEFunctions.h>
// #include <arm_compute/runtime/Tensor.h>
// #include <arm_compute/core/Types.h>

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

void cublas_gemm_template(size_t n_iter, size_t n_bytes, size_t a_target, size_t b_target, size_t c_target, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(float);
    size_t matrix_side = std::sqrt(n_elems);

    MmapDataFactory A(n_elems * sizeof(float));
    MmapDataFactory B(n_elems * sizeof(float));
    MmapDataFactory C(n_elems * sizeof(float));
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

// void arm_compute_gemm_template(size_t n_iter, size_t n_bytes, size_t a_target, size_t b_target, size_t c_target, std::string name) {
//     double times[n_iter];
//     size_t n_elems = n_bytes / sizeof(float);
//     size_t matrix_side = std::sqrt(n_elems);

//     MmapDataFactory A(n_elems * sizeof(float));
//     MmapDataFactory B(n_elems * sizeof(float));
//     MmapDataFactory C(n_elems * sizeof(float));
//     dispatch_command(a_target, WRITE, A.data, n_elems * sizeof(float));
//     dispatch_command(b_target, WRITE, B.data, n_elems * sizeof(float));
//     dispatch_command(c_target, WRITE, C.data, n_elems * sizeof(float));

//     fill_random((float *) A.data, n_elems * sizeof(float));
//     fill_random((float *) B.data, n_elems * sizeof(float));
    
//     arm_compute::Tensor tA;
//     arm_compute::Tensor tB;
//     arm_compute::Tensor tC;

//     const arm_compute::TensorShape shapeA(matrix_size, matrix_size);
//     const arm_compute::TensorShape shapeB(matrix_size, matrix_size);
//     const arm_compute::TensorShape shapeC(matrix_size, matrix_size);

//     // Initialize arm_compute::tensors with the defined shapes
//     tA.allocator()->init(arm_compute::TensorInfo(shapeA, 1, DataType::F32));
//     tB.allocator()->init(arm_compute::TensorInfo(shapeB, 1, DataType::F32));
//     tC.allocator()->init(arm_compute::TensorInfo(shapeC, 1, DataType::F32));

//     NEONScheduler::get().default_init();
//     arm_compute::NEGEMM gemm;
//     gemm.configure(&tA, &tB, nullptr, &tC, 1.0f, 0.0f);


//     for (size_t iter = 0; iter < n_iter; ++iter) {
//         uint64_t start = get_cpu_clock();
//         gemm.run();
//         uint64_t end = get_cpu_clock();
//         times[iter] = get_elapsed_milliseconds_clock(start, end);
//     }

//     millisecond_times_to_gb_sec_file(times, n_iter, n_bytes, "results/apps/gemm/arm_compute/" + name);
// }

void run_cublas_gemm_tests(size_t n_iter, size_t n_bytes) {
    cublas_gemm_template(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr/" + std::to_string(n_bytes)); // ddr uniform
    cublas_gemm_template(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, DEVICE_ID, "hbm/" + std::to_string(n_bytes)); // hbm uniform
    cublas_gemm_template(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_hbm_ddr/" + std::to_string(n_bytes)); // hbm -> ddr
    cublas_gemm_template(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_ddr_hbm/" + std::to_string(n_bytes)); // ddr->hbm
    cublas_gemm_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_ddr/" + std::to_string(n_bytes)); // het->ddr
    cublas_gemm_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_ddr_hbm/" + std::to_string(n_bytes)); // het->hbm
}

// void run_arm_compute_gemm_tests(size_t n_iter, size_t n_bytes) {
//     arm_compute_gemm_template(n_iter, n_bytes, HOST_ID, HOST_ID, HOST_ID, "ddr/" + std::to_string(n_bytes)); // ddr uniform
//     arm_compute_gemm_template(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, DEVICE_ID, "hbm/" + std::to_string(n_bytes)); // hbm uniform
//     arm_compute_gemm_template(n_iter, n_bytes, DEVICE_ID, DEVICE_ID, HOST_ID, "hbm_hbm_ddr/" + std::to_string(n_bytes)); // hbm -> ddr
//     arm_compute_gemm_template(n_iter, n_bytes, HOST_ID, HOST_ID, DEVICE_ID, "ddr_ddr_hbm/" + std::to_string(n_bytes)); // ddr->hbm
//     arm_compute_gemm_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, HOST_ID, "hbm_ddr_ddr/" + std::to_string(n_bytes)); // het->ddr
//     arm_compute_gemm_template(n_iter, n_bytes, DEVICE_ID, HOST_ID, DEVICE_ID, "hbm_ddr_hbm/" + std::to_string(n_bytes)); // het->hbm
// }

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

template <typename Policy>
void thrust_reduction_template(size_t n_iter, size_t n_bytes, size_t target, const Policy &exec, std::string name) {
    double times[n_iter];
    size_t n_elems = n_bytes / sizeof(float);

    MmapDataFactory factory(n_elems * sizeof(float));
    dispatch_command(target, WRITE, factory.data, n_elems * sizeof(float));
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
    thrust_reduction_template<>(n_iter, n_bytes, HOST_ID, thrust::host, "host/ddr/" + std::to_string(n_bytes));
    thrust_reduction_template<>(n_iter, n_bytes, DEVICE_ID, thrust::host, "host/hbm/" + std::to_string(n_bytes));
    thrust_reduction_template<>(n_iter, n_bytes, HOST_ID, thrust::device, "device/ddr/" + std::to_string(n_bytes));
    thrust_reduction_template<>(n_iter, n_bytes, DEVICE_ID, thrust::device, "device/hbm/" + std::to_string(n_bytes));
}