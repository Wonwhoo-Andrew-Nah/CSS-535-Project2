// $ nvcc -o output example.cu -lcufft

#include <stdio.h>
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

int main()
{       
        // length of the signal
        const int N = 16;
        const int blockSize = 8;

        // Host memory
        cuDoubleComplex *h_signal = new cuDoubleComplex[N];

        // input data initialization
        for (int i = 0; i < N; ++i){
                h_signal[i].x = sin(2 * PI * i / N); // real
                h_signal[i].y = 0; // imaginary
        }

        // Copy to device
        cuDoubleComplex *d_signal;
        CUDA_CHECK(cudaMalloc((void**)&d_signal, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_signal, h_signal, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        // call kernel function
        perform_fft << < (N + blockSize - 1) / blockSize, blockSize >> > (d_signal, N);
        
        // Copy to host
        cuDoubleComplex *h_result = new cuDoubleComplex[N];
        CUDA_CHECK(cudaMemcpy(h_result, d_signal, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // check the results
        std::cout << "unoptimized" << std::endl;
        for (int i = 0; i< N; ++i){
                std::cout<< "(" << h_result[i].x << ", " << h_result[i].y << ")" << std::endl;
        }

        // deallocate memory
        delete[] h_signal;
        delete[] h_result;
        CUDA_CHECK(cudaFree(d_signal));

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
}
