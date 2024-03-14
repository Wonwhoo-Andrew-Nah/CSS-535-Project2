#include <cuComplex.h>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void perform_fft(cuDoubleComplex* signal, int n);
void test_our_fft();

#pragma once
