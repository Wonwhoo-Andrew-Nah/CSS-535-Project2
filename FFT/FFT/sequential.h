#pragma once
#ifndef SEQUENTIAL_FFT_H
#define SEQUENTIAL_FFT_H

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

typedef std::complex<double> Complex;

void sequential_fft(std::vector<Complex>& input);
void sequential_ifft(std::vector<Complex>& input);

#endif