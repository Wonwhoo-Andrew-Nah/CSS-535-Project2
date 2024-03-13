#pragma once
#ifndef SEQUENTIAL_FFT_H
#define SEQUENTIAL_FFT_H

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

struct Complex {
    double x, y;
    Complex() : x(0), y(0) {}
    Complex(double real, double imaginary) : x(real), y(imaginary) {}
};

void sequential_fft(std::vector<Complex>& input);

#endif