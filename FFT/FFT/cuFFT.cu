// compile with following prompt
// $ nvcc -o output example.cu -lcufft

#include <stdio.h>
#include <vector>
#include <math.h>
#include <cuComplex.h>
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846
#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << "at" << file << ":" << line << std::endl;
        exit(-1);
    }
}

int main() {
    
    std::vector<int> input_sizes = { 2<<1, 2<<3, 2<<4, 2<<6, 2<<8, 2<<16, 2<<20 };

    for (int N : input_sizes) {
        // Host memory
        cuDoubleComplex* h_data = new cuDoubleComplex[N];
        for (int i = 0; i < N; ++i) {
            h_data[i].x = sin(2 * PI * i / N); // real
            h_data[i].y = 0; // imaginary
        }

        // Copy to Device
        cuDoubleComplex* d_data;
        CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        // Instantiate cuFFT plan
        cufftHandle plan;
        cufftResult_t result = cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "cuFFT plan creation failed" << std::endl;
            exit(-1);
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Start timing
        cudaEventRecord(start);

        // Run cuFFT
        result = cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "cuFFT execution failed" << std::endl;
            exit(-1);
        }

        // End timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Print the results
        std::cout << "Input size: " << N << std::endl;
        std::cout << "cuFFT execution time: " << milliseconds << " ms" << std::endl;

        // Cleanup
        cufftDestroy(plan);
        CUDA_CHECK(cudaFree(d_data));
        delete[] h_data;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return 0;
}
