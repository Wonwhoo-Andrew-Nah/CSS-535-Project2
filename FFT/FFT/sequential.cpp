#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

typedef std::complex<double> Complex;

// Recursive FFT implementation
void sequential_fft(std::vector<Complex>& input) {
    const size_t N = input.size();
    if (N <= 1) return;

    std::vector<Complex> even(N / 2), odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    sequential_fft(even);
    sequential_fft(odd);

    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * 3.14159265358979323846 * k / N) * odd[k];
        input[k] = even[k] + t;
        input[k + N / 2] = even[k] - t;
    }
}

// Inverse FFT
void sequential_ifft(std::vector<Complex>& input) {
    // conjugate the complex numbers
    for (auto& val : input)
        val = std::conj(val);

    // forward FFT
    sequential_fft(input);

    // conjugate the complex numbers again and scale
    for (auto& val : input) {
        val = std::conj(val) / static_cast<double>(input.size());
    }
}