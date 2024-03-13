//#include <stdio.h>
//#include <math.h>
//#include <cuComplex.h>
//#include <iostream>
//#include <cufft.h>
//#include <cuda_runtime.h>
//
//#define PI 3.14159265358979323846
//
//void checkCudaError(cudaError_t error, const char* file, int line) {
//    if (error != cudaSuccess) {
//        std::cerr << "CUDA error: " << cudaGetErrorString(error) << "at" << file << ":" << line << std::endl;
//        exit(-1);
//    }
//}
//
//__device__ void fft(cuDoubleComplex* signal, int n, int offset, int step)
//{
//    if (n <= 1) return;
//
//    cuDoubleComplex even = signal[offset];
//    cuDoubleComplex odd = signal[offset + step];
//
//    fft(signal, n / 2, offset, step * 2);
//    fft(signal, n / 2, offset + step, step * 2);
//
//    for (int i = 0; i < n / 2; i++)
//    {
//        double angle = -2 * PI * i / n;
//
//        cuDoubleComplex temp;
//        temp.x = cos(angle) * cuCreal(odd) - sin(angle) * cuCimag(odd);
//        temp.y = cos(angle) * cuCimag(odd) + sin(angle) * cuCreal(odd);
//
//        signal[offset + i] = cuCadd(even, temp);
//        signal[offset + i + n / 2] = cuCsub(even, temp);
//
//        even = signal[offset + i + step];
//    }
//}
//
//__global__ void perform_fft(cuDoubleComplex* signal, int n)
//{
//    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//
//    fft(signal, n, threadID, 1);
//}
//
//int main() {
//    
//    int N = 10;
//
//    return 0;
//}
