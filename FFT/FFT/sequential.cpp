#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

#define PI 3.14159265358979323846

struct Complex {
    double x, y;
    Complex() : x(0), y(0) {}
    Complex(double real, double imaginary) : x(real), y(imaginary) {}
};

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
        double angle = -2 * PI * k / N;
        Complex t(cos(angle), sin(angle));
        Complex temp;
        temp.x = t.x * odd[k].x - t.y * odd[k].y;
        temp.y = t.x * odd[k].y + t.y * odd[k].x;
        input[k] = Complex(even[k].x + temp.x, even[k].y + temp.y);
        input[k + N / 2] = Complex(even[k].x - temp.x, even[k].y - temp.y);
    }
}

int main() {

    const int N = 16;
    std::vector<Complex> h_data(N);

    for (int i = 0; i < N; ++i) {
        h_data[i].x = i; // real
        h_data[i].y = 0; // imaginary
    }

    sequential_fft(h_data);

    std::cout << "sequential result:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "(" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

    return 0;
}
