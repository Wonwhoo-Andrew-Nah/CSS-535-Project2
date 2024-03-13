// Forward functions here.
#include "sequential.h"
#include "parallelized.h"
#include "cuFFT.h"
#include "parallelized_unoptimized.h"

int main() {

    // std::vector<Complex> data = { 1, 2, 3, 4, 5, 6, 7, 8 };
    // std::string musicPath = "Assets/01 A Head Full Of Dreams.m4a";
    // sequential_fft(data);
    // sequential_ifft(data);

	// parallelized();
	// cuFFT(data);
    // file_cuFFT(musicPath);

    test_our_fft();

    // Printing the result
    // for (const auto& val : data) {
    //    std::cout << val << std::endl;
    // }

	return 0;
}