#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846
#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

__device__ int bitReverse(int n, int bits) {
    int reversedN = n;
    int count = bits - 1;

    n >>= 1;
    while (n > 0) {
        reversedN = (reversedN << 1) | (n & 1);
        count--;
        n >>= 1;
    }
    return ((reversedN << count) & ((1 << bits) - 1));
}

__global__ void perform_fft(cuDoubleComplex* signal, int n, int log2n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int reversed = bitReverse(tid, log2n);
    if (reversed > tid) {
        cuDoubleComplex temp = signal[tid];
        signal[tid] = signal[reversed];
        signal[reversed] = temp;
    }

    __syncthreads();

    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        cuDoubleComplex w_m = make_cuDoubleComplex(cos(-2.0 * PI / m), sin(-2.0 * PI / m));

        int j = tid & (m2 - 1);
        if (j < m2) {
            cuDoubleComplex w = make_cuDoubleComplex(1.0, 0.0);
            for (int k = 0; k < j; ++k) {
                w = cuCmul(w, w_m);
            }

            int partner = tid + m2;
            cuDoubleComplex t = cuCmul(w, signal[partner]);
            signal[partner] = cuCsub(signal[tid], t);
            signal[tid] = cuCadd(signal[tid], t);
        }
        __syncthreads();
    }
}

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << file << ":" << line << std::endl;
        exit(-1);
    }
}


int main() {

	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));

    std::vector<int> input_sizes = { 2<<1, 2<<3, 2<<4, 2<<6, 2<<8, 2<<16};
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

        cudaEventRecord(start);
        perform_fft <<<(N + blockSize - 1) / blockSize, blockSize>>> (d_signal, N, log2(N));
        cudaEventRecord(stop);

        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        std::cout << "Input size: " << N << std::endl;
        std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

        // cuDoubleComplex *h_result = new cuDoubleComplex[N];
        // CUDA_CHECK(cudaMemcpy(h_result, d_signal, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // for (int i = 0; i< N; ++i) {
        //     std::cout << "(" << h_result[i].x << ", " << h_result[i].y << ")" << std::endl;
        // }
        // delete[] h_result;

        delete[] h_signal;
        CUDA_CHECK(cudaFree(d_signal));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return 0;
}
