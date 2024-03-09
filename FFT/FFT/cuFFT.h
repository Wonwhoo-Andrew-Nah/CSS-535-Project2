#pragma once
#ifndef FFT_H
#define FFT_H

#include <iostream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>

#include <fstream>
#include <string>

typedef std::complex<double> Complex;

void cuFFT(std::vector<Complex>& input);
std::vector<Complex> readDataFromFile(const std::string& filename);
void file_cuFFT(std::string filename);

#endif
