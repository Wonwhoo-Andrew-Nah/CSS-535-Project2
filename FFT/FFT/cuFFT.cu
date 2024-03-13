// compile with following prompt
// $ nvcc -o output example.cu -lcufft

#include <stdio.h>
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
    
    int N = 16;

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

    // run cuFFT
    result = cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD); // Run Forward FFT
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT execution failed" << std::endl;
        exit(-1);
    }

    // print the results
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    std::cout << "cuFFT :" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "(" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

    // free memory
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;

    return 0;

    // unoptimized
    // (0.382683, 0)
    // (1.44334, -0.146447)
    // (2.55487, -0.541196)
    // (3.42388, -1.20711)
    // (3.77164, -2.07193)
    // (3.39875, -2.97487)
    // (2.23784, -3.69552)
    // (0.382683, -4)
    // (-1.53073, -3.69552)
    // (-1.03153, -2.82843)
    // (0.165911, -2.07193)
    // (1.63099, -1.70711)
    // (2.84776, -1.84776)
    // (3.33809, -2.41421)
    // (2.77904, -3.15432)
    // (-0.382683, 4)

    //Z2Z
    // (6.69535e-17, 0)
    // (-1.30502e-15, -8)
    // (-3.87815e-16, -7.32325e-16)
    // (1.19873e-16, -6.48535e-16)
    // (1.22465e-16, -4.996e-16)
    // (1.19873e-16, -6.48535e-16)
    // (6.32745e-16, -2.88235e-16)
    // (5.75415e-16, 0)
    // (1.77976e-16, 0)
    // (5.75415e-16, 0)
    // (6.32745e-16, 2.88235e-16)
    // (1.19873e-16, 6.48535e-16)
    // (1.22465e-16, 4.996e-16)
    // (1.19873e-16, 6.48535e-16)
    // (-3.87815e-16, 7.32325e-16)
    // (-1.30502e-15, 8)
}
