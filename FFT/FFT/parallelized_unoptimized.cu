#include "parallelized_unoptimized.h"

#define PI 3.14159265358979323846

__device__ void fft(cuDoubleComplex* signal, int n, int offset, int step) {
	if (n <= 1) return;

	cuDoubleComplex even = signal[offset];
	cuDoubleComplex odd = signal[offset + step];

	fft(signal, n / 2, offset, step * 2);
	fft(signal, n / 2, offset + step, step * 2);

	for (int i = 0; i < n / 2; i++)
	{
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

void test_our_fft() {
	const int N = 40;
	cuDoubleComplex* signal;
	cuDoubleComplex* d_signal;
	signal = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
	for (int i = 0; i < N; i++) {
		signal[i] = make_cuDoubleComplex(sin(2 * PI * i / N), cos(2 * PI * i / N));
	}
	cudaMalloc(&d_signal, N * sizeof(cuDoubleComplex));
	cudaMemcpy(d_signal, signal, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	perform_fft << <1, N >> > (d_signal, N);
	cudaMemcpy(signal, d_signal, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	printf("FFT Results:\n");
	for (int i = 0; i < N; i++) {
		printf("(%f, %f)\n", cuCreal(signal[i]), cuCimag(signal[i]));
	}
	cudaFree(d_signal);
	free(signal);
}
