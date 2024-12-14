#include "lanczos_resample.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cmath>

using namespace std;

__device__ double lanczos_kernel(double x, int a) {
    if (x == 0.0) return 1.0;
    if (x <= -a || x >= a) return 0.0;
    double pix = PI * x;
    return (sin(pix) * sin(pix / a)) / (pix * pix / a);
}



// CUDA kernel for Lanczos resampling
__global__ void lanczos_resample_kernel(
    const unsigned char* input,
    unsigned char* output,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    int channels,
    int a  // Include 'a' parameter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Output pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Output pixel y

    if (x >= out_width || y >= out_height)
        return;

    int idx_out = (y * out_width + x) * channels;

    double sum[3] = { 0.0, 0.0, 0.0 };
    double sum_weight = 0.0;

    // Compute the corresponding input coordinate
    double scale_x = static_cast<double>(out_width) / in_width;
    double scale_y = static_cast<double>(out_height) / in_height;
    double src_x = (x + 0.5) / scale_x - 0.5;
    double src_y = (y + 0.5) / scale_y - 0.5;

    // Compute the window boundaries
    int x_start = static_cast<int>(floor(src_x - a + 1));
    int x_end = static_cast<int>(floor(src_x + a));
    int y_start = static_cast<int>(floor(src_y - a + 1));
    int y_end = static_cast<int>(floor(src_y + a));

    x_start = max(0, x_start);
    x_end = min(in_width - 1, x_end);
    y_start = max(0, y_start);
    y_end = min(in_height - 1, y_end);

    for (int j = y_start; j <= y_end; ++j) {
        double dist_y = src_y - j;
        double wy = lanczos_kernel(dist_y, a);
        for (int i = x_start; i <= x_end; ++i) {
            double dist_x = src_x - i;
            double wx = lanczos_kernel(dist_x, a);
            double weight = wx * wy;

            int idx_in = (j * in_width + i) * channels;

            for (int c = 0; c < channels; ++c) {
                sum[c] += weight * input[idx_in + c];
            }
            sum_weight += weight;
        }
    }

    if (sum_weight > 0.0) {
        for (int c = 0; c < channels; ++c) {
            sum[c] /= sum_weight;
            sum[c] = fmin(fmax(sum[c], 0.0), 255.0);
            output[idx_out + c] = static_cast<unsigned char>(sum[c] + 0.5);
        }
    }
    else {
        for (int c = 0; c < channels; ++c) {
            output[idx_out + c] = 0;
        }
    }
}



// Structure to hold device copies of AxisParam data
//struct DeviceAxisParam {
//    int* start;
//    int* length;
//    double* weight;
//    int* index;
//
//    explicit DeviceAxisParam(const AxisParam& param) {
//        cudaMalloc(&start, sizeof(int) * param.start.size());
//        cudaMalloc(&length, sizeof(int) * param.length.size());
//        cudaMalloc(&weight, sizeof(double) * param.weight.size());
//        cudaMalloc(&index, sizeof(int) * param.index.size());
//
//        cudaMemcpy(start, param.start.data(), sizeof(int) * param.start.size(), cudaMemcpyHostToDevice);
//        cudaMemcpy(length, param.length.data(), sizeof(int) * param.length.size(), cudaMemcpyHostToDevice);
//        cudaMemcpy(weight, param.weight.data(), sizeof(double) * param.weight.size(), cudaMemcpyHostToDevice);
//        cudaMemcpy(index, param.index.data(), sizeof(int) * param.index.size(), cudaMemcpyHostToDevice);
//    }
//
//    ~DeviceAxisParam() {
//        cudaFree(start);
//        cudaFree(length);
//        cudaFree(weight);
//        cudaFree(index);
//    }
//};

// Function to perform Lanczos resampling on the GPU
// Function to perform Lanczos resampling on the GPU
void lanczos_resample_cuda(
    const unsigned char* h_input,
    unsigned char* h_output,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    int channels,
    int a)
{
    // Allocate device memory for input and output images
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    size_t input_size = in_width * in_height * channels * sizeof(unsigned char);
    size_t output_size = out_width * out_height * channels * sizeof(unsigned char);

    cudaError_t err;

    err = cudaMalloc(&d_input, input_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for input: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_output, output_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return;
    }

    // Copy input image to device
    err = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for input: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x,
                  (out_height + blockSize.y - 1) / blockSize.y);

    lanczos_resample_kernel<<<gridSize, blockSize>>>(
        d_input,
        d_output,
        in_width,
        in_height,
        out_width,
        out_height,
        channels,
        a);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronize error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Copy output image back to host
    err = cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

