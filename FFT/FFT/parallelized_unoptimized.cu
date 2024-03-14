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

__device__ void fft(cuDoubleComplex* signal, int n, int offset, int step) {
	if (n <= 1) return;

	cuDoubleComplex even = signal[offset];
	cuDoubleComplex odd = signal[offset + step];

	fft(signal, n / 2, offset, step * 2);
	fft(signal, n / 2, offset + step, step * 2);

	for (int i = 0; i < n / 2; i++) {
		double angle = -2 * PI * i / n;

		cuDoubleComplex temp;
		temp.x = cos(angle) * cuCreal(odd) - sin(angle) * cuCimag(odd);
		temp.y = cos(angle) * cuCimag(odd) + sin(angle) * cuCreal(odd);

		signal[offset + i] = cuCadd(even, temp);
		signal[offset + i + n / 2] = cuCsub(even, temp);

		even = signal[offset + i + step];
	}
}

__global__ void perform_fft(cuDoubleComplex* signal, int n) {
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	fft(signal, n, threadID, 1);
}

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << "at" << file << ":" << line << std::endl;
        exit(-1);
    }
}

int main() {

	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));

    std::vector<int> input_sizes = { 2<<1, 2<<3, 2<<4, 2<<6, 2<<8, 2<<16, 2<<20 };
    const int blockSize = 256;

    for (int N : input_sizes) {

        cuDoubleComplex* h_signal = new cuDoubleComplex[N];
        for (int i = 0; i < N; ++i) {
            h_signal[i].x = sin(2 * PI * i / N); // real
            h_signal[i].y = 0; // imaginary
        }

        cuDoubleComplex* d_signal;
        CUDA_CHECK(cudaMalloc((void**)&d_signal, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_signal, h_signal, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        perform_fft <<<(N + blockSize - 1) / blockSize, blockSize>>> (d_signal, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        std::cout << "Input size: " << N << std::endl;
        std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

        cuDoubleComplex *h_result = new cuDoubleComplex[N];
        CUDA_CHECK(cudaMemcpy(h_result, d_signal, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // for (int i = 0; i< 4; ++i) {
        //     std::cout << "(" << h_result[i].x << ", " << h_result[i].y << ")" << std::endl;
        // }

        delete[] h_result;
        delete[] h_signal;
        CUDA_CHECK(cudaFree(d_signal));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return 0;
}
