// Forward functions here.
#include "sequential.h"
#include "parallelized.h"
#include "cuFFT.h"
#include "parallelized_unoptimized.h"

struct Complex {
    double x, y;
    Complex() : x(0), y(0) {}
    Complex(double real, double imaginary) : x(real), y(imaginary) {}
};

void runCuFFT(Complex* input, int size) {
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_Z2Z, 1);
    Complex* d_data;
    cudaMalloc((void**)&d_data, sizeof(Complex) * size);
    cudaMemcpy(d_data, input, sizeof(Complex) * size, cudaMemcpyHostToDevice);

    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_data, (cufftDoubleComplex*)d_data, CUFFT_FORWARD);
    cudaMemcpy(input, d_data, sizeof(Complex) * size, cudaMemcpyDeviceToHost);

    // Result processing and cleanup code here

}

#define PI 3.14159265358979323846

void test_cuFFT() {
    const int N = 99;
    cuDoubleComplex* signal;
    cuDoubleComplex* d_signal;
    signal = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    for (int i = 0; i < N; i++) {
        signal[i] = make_cuDoubleComplex(sin(2 * PI * i / N), cos(2 * PI * i / N));
    }
    cudaMalloc(&d_signal, N * sizeof(cuDoubleComplex));
    cudaMemcpy(d_signal, signal, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_signal, (cufftDoubleComplex*)d_signal, CUFFT_FORWARD);
    cudaMemcpy(signal, d_signal, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    printf("cuFFT Results:\n");
    for (int i = 0; i < N; i++) {
        printf("(%f, %f)\n", cuCreal(signal[i]), cuCimag(signal[i]));
    }
    cufftDestroy(plan);
    cudaFree(d_signal);
    free(signal);
}

int main() {

    const int N = 16;
    std::vector<Complex> h_data(N);

    for (int i = 0; i < N; ++i) {
        h_data[i].x = sin(2 * PI * i / N); // real
        h_data[i].y = 0; // imaginary
    }
    sequential_fft(h_data);

    std::cout << "FFT result:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "(" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

	// parallelized();
	// cuFFT(data);
    // file_cuFFT(musicPath);

    test_cuFFT();

    // Printing the result
    // for (const auto& val : data) {
    //    std::cout << val << std::endl;
    // }

	return 0;
}