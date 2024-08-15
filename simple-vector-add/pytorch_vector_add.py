# Vector addition implementation with Python

import torch
import time
import matplotlib.pyplot as plt

vector_sizes = [1024, 1024*1024, 1024*1024*10, 1024*1024*100]
gpu_times = []
cpu_times = []

for size in vector_sizes:
    # GPU time
    vector1 = torch.sin(torch.arange(size, dtype=torch.float32)).cuda()
    vector2 = torch.cos(torch.arange(size, dtype=torch.float32)).cuda()

    start_time = time.time()
    result = vector1 + vector2
    end_time = time.time()
    gpu_times.append(end_time - start_time)

    # CPU time
    vector1 = torch.sin(torch.arange(size, dtype=torch.float32))
    vector2 = torch.cos(torch.arange(size, dtype=torch.float32))

    start_time = time.time()
    result = vector1 + vector2
    end_time = time.time()
    cpu_times.append(end_time - start_time)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(vector_sizes, gpu_times, label='GPU', marker='o')
plt.plot(vector_sizes, cpu_times, label='CPU', marker='o')
plt.xlabel('Vector Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Vector Size')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)

for i, (size, time) in enumerate(zip(vector_sizes, gpu_times)):
    plt.text(size, time, f'{time:.6f}', ha='center', va='bottom')
for i, (size, time) in enumerate(zip(vector_sizes, cpu_times)):
    plt.text(size, time, f'{time:.6f}', ha='center', va='bottom')

# Save plot if plt.show() does not work
plt.savefig('pytorch_execution_time_vs_vector_size.png')
print("Plot saved as 'pytorch_execution_time_vs_vector_size.png'")
plt.show()
