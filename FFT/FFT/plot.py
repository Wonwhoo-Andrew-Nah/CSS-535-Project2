import matplotlib.pyplot as plt

# data
input_sizes = [4, 32, 128, 512, 131072]
sequential_fft = [0.098, 0.0372, 0.1524, 0.6986, 258.262]
parallel_naive_dft = [0.475136, 0.07168, 0.274432, 1.08032, 243.621]
parallel_naive_fft = [0.20992, 0.017408, 0.044032, 0.17408, 625.194]
parallel_optimized_fft = [0.247808, 0.019456, 0.033792, 0.07168, 3.82787]
cufft = [0.203776, 0.027648, 0.026624, 0.036864, 0.252928]

# draw
plt.figure(figsize=(10, 6))
plt.plot(input_sizes, sequential_fft, label='Sequential FFT', marker='o')
plt.plot(input_sizes, parallel_naive_dft, label='Parallelized naïve DFT', marker='o')
plt.plot(input_sizes, parallel_naive_fft, label='Parallelized naïve FFT', marker='o')
plt.plot(input_sizes, parallel_optimized_fft, label='Parallelized optimized FFT', marker='o')
plt.plot(input_sizes, cufft, label='cuFFT', marker='o')
plt.xticks(input_sizes, input_sizes)
plt.xscale('log')
# tile and lable
plt.title('Execution time by input sizes')
plt.xlabel('Input size')
plt.ylabel('Execution time (ms)')
plt.legend()

# show
plt.grid(True)
plt.savefig('execution_time_plot.png')
