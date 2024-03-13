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
        h_data[i].x = i; // real
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

    //unoptimized
    //(1, 0)
    //(3.92388, -0.382683)
    //(7.55487, -1.47247)
    //(11.5685, -3.48614)
    //(15.5822, -6.49981)
    //(19.2132, -10.4374)
    //(22.1371, -15.082)
    //(24.1371, -20.1094)
    //(8.13707, -25.1367)
    //(2.89828, -26.3372)
    //(-0.325033, -25.0588)
    //(-0.737654, -22.2616)
    //(2.02771, -19.1367)
    //(7.85455, -16.9252)
    //(16.1602, -16.7293)
    //(25.9848, -19.344)

    //Z2Z
    //(120, 0)
    //(-8, 40.2187)
    //(-8, 19.3137)
    //(-8, 11.9728)
    //(-8, 8)
    //(-8, 5.34543)
    //(-8, 3.31371)
    //(-8, 1.5913)
    //(-8, 0)
    //(-8, -1.5913)
    //(-8, -3.31371)
    //(-8, -5.34543)
    //(-8, -8)
    //(-8, -11.9728)
    //(-8, -19.3137)
    //(-8, -40.2187)
}
