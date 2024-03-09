#pragma comment(lib, "cufft.lib")

#include <iostream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>

#include <fstream>
#include <string>

typedef std::complex<double> Complex;

__global__ void cuFFT(std::vector<Complex>& input) {
	
	// initialize CUDA
	cudaSetDevice(0);

	// create cuFFT handle
	cufftHandle plan;
	cufftPlan1d(&plan, input.size(), CUFFT_Z2Z, 1);

	// host to device
	Complex* d_data;
	cudaMalloc((void**)&d_data, sizeof(Complex) * input.size());
    cudaMemcpy(d_data, input.data(), sizeof(Complex) * input.size(), cudaMemcpyHostToDevice);

    // run FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_data, (cufftDoubleComplex*)d_data, CUFFT_FORWARD);

    // device to host
    std::vector<Complex> result(input.size());
    cudaMemcpy(result.data(), d_data, sizeof(Complex) * input.size(), cudaMemcpyDeviceToHost);

    // print the result
    std::cout << "Result of cuFFT:" << std::endl;
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << std::endl;
    }

    // Free memory
    cudaFree(d_data);
    cufftDestroy(plan);
}

// read from file
std::vector<Complex> readDataFromFile(const std::string& filename) {
    std::ifstream inputFile(filename, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Calculate the file size
    inputFile.seekg(0, std::ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    // Make file data
    std::vector<Complex> data(fileSize / sizeof(Complex));

    // read data from file
    inputFile.read(reinterpret_cast<char*>(data.data()), fileSize);
    inputFile.close();

    return data;
}

__global__ void file_cuFFT(std::string filename) {

    std::vector<Complex> input = readDataFromFile(filename);

    // initialize CUDA
    cudaSetDevice(0);

    // create cuFFT handle
    cufftHandle plan;
    cufftPlan1d(&plan, input.size(), CUFFT_Z2Z, 1);

    // host to device
    Complex* d_data;
    cudaMalloc((void**)&d_data, sizeof(Complex) * input.size());
    cudaMemcpy(d_data, input.data(), sizeof(Complex) * input.size(), cudaMemcpyHostToDevice);

    // run FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_data, (cufftDoubleComplex*)d_data, CUFFT_FORWARD);

    // device to host
    std::vector<Complex> result(input.size());
    cudaMemcpy(result.data(), d_data, sizeof(Complex) * input.size(), cudaMemcpyDeviceToHost);

    // print the result
    std::cout << "Result of cuFFT:" << std::endl;
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << std::endl;
    }

    // Free memory
    cudaFree(d_data);
    cufftDestroy(plan);
}