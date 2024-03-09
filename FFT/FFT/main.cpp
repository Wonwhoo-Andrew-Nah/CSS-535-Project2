#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

// Forward functions here.
#include "sequential.h"
#include "parallelized.h"
#include "cuFFT.h"

int main() {

    std::vector<Complex> data = { 1, 2, 3, 4, 5, 6, 7, 8 };
	parallelized();
	cuFFT();

    // Forward FFT
    sequential_fft(data);

    // Inverse FFT
    sequential_ifft(data);

    // Printing the result
    for (const auto& val : data) {
        std::cout << val << std::endl;
    }

	return 0;
}