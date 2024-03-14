#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cuComplex.h>
#include <chrono>

#define PI 3.14159265358979323846
#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << "at" << file << ":" << line << std::endl;
        exit(-1);
    }
}

__device__ cuDoubleComplex w(int N, int k) {
    return make_cuDoubleComplex(cos(-2 * PI * k / N), sin(-2 * PI * k / N));
}

__global__ void fftKernel(cuDoubleComplex *data, int N, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 2) {
        int k = idx + offset;
        cuDoubleComplex W = w(N, k);
        cuDoubleComplex even = data[2 * idx];
        cuDoubleComplex odd = cuCmul(W, data[2 * idx + 1]);
        data[idx] = cuCadd(even, odd);
        data[idx + N / 2] = cuCsub(even, odd);
    }
}

int main() {

    std::vector<int> input_sizes = { 2<<1, 2<<3, 2<<4, 2<<6, 2<<8, 2<<16, 2<<20 };

    for (int N : input_sizes) {
        
        cuDoubleComplex* h_signal = new cuDoubleComplex[N];
        for (int i = 0; i < N; ++i) {
            // h_data[i].x = sin(2 * PI * i / N); // real
            h_signal[i].x = i; // real
            h_signal[i].y = 0; // imaginary
        }

        cuDoubleComplex* d_signal;
        CUDA_CHECK(cudaMalloc((void**)&d_signal, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_signal, h_signal, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int size = 2; size <= N; size <<= 1) {
            int numBlocks = (N / size) / 256 + 1;
            fftKernel<<<numBlocks, 256>>>(d_signal, N, N / size);
        }   
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        std::cout << "Input size: " << N << std::endl;
        std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

        cuDoubleComplex *h_result = new cuDoubleComplex[N];
        CUDA_CHECK(cudaMemcpy(h_result, d_signal, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // print first 16 results
        std::cout << "First 4 elements of the result:" << std::endl;
        for (int i = 0; i < 4; ++i) {
            std::cout << "(" << h_result[i].x << ", " << h_result[i].y << ") " << std::endl;
        }
        std::cout << std::endl;

        delete[] h_signal;
        delete[] h_result;
        CUDA_CHECK(cudaFree(d_signal));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return 0;
}
