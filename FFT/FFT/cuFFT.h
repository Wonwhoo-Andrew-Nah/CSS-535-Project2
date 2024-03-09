#pragma once
#ifndef FFT_H
#define FFT_H

#include <iostream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>

typedef std::complex<double> Complex;

void cuFFT(std::vector<Complex>& input);

#endif
