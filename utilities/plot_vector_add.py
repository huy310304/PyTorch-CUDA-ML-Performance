import matplotlib.pyplot as plt
import numpy as np

# Data from the provided images
vector_size = [10**3, 10**6, 10**7, 10**8]

# CUDA Data
gpu_cuda_time = [0.000062, 0.000034, 0.000063, 0.000063]

# PyTorch Data
cpu_pytorch_time = [0.000024, 0.000667, 0.004139, 0.042366]
gpu_pytorch_time = [0.000135, 0.000042, 0.000304, 0.002363]

# Plotting
plt.figure(figsize=(10, 6))

# CUDA
plt.plot(vector_size, gpu_cuda_time, label='GPU (CUDA)', marker='o', linestyle='-', color='blue')

# PyTorch
plt.plot(vector_size, gpu_pytorch_time, label='GPU (PyTorch)', marker='x', linestyle='--', color='blue')
plt.plot(vector_size, cpu_pytorch_time, label='CPU (PyTorch)', marker='x', linestyle='--', color='orange')

# Logarithmic scale
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel('Vector Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Vector Size: CUDA vs PyTorch')
plt.legend()

# Show the plot
plt.show()
