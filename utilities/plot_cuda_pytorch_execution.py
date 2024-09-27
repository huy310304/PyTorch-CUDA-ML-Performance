# Re-importing the necessary packages after reset
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting (updated)
points = [500, 5000, 50000, 500000, 5000000, 10000000, 50000000, 100000000]
pytorch_cpu_updated = [0.0876, 0.181, 0.29637, 1.89209, 9.77955, 18.43, 112.13, 206.55]
pytorch_gpu_updated = [0.52, 0.51, 0.65, 0.67, 1.00, 1.24, 3.98, 7.38]
cuda_shared_gpu_updated = [0.0038684, 0.0039203, 0.0043834, 0.0094840, 0.089, 0.339999, 1.43946, 2.8308]

# Create plot with updated data and bolder, larger markers
plt.figure(figsize=(10, 6))

# Plot each dataset with larger, bolder markers
plt.plot(points, pytorch_cpu_updated, label='PyTorch CPU', marker='x', markersize=10, markeredgewidth=2)
plt.plot(points, pytorch_gpu_updated, label='PyTorch GPU', marker='d', markersize=10, markeredgewidth=2)
plt.plot(points, cuda_shared_gpu_updated, label='CUDA Shared Memory Optimization GPU', marker='s', color='red', markersize=10, markeredgewidth=2)

# Set log scale
plt.xscale('log')
plt.yscale('log')

# Set xticks and yticks
plt.xticks([500, 5000, 50000, 500000, 5000000, 10000000, 50000000, 100000000], ['500', '5K', '50K', '500K', '5M', '10M', '50M', '100M'], fontsize=18)
plt.yticks([0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1000'], fontsize=20)

# Set labels and title
plt.xlabel('DataPoints', fontsize=20)
plt.ylabel('Time (s)', fontsize=20)
plt.title('', fontsize=15)

# Show legend with larger font size
plt.legend(fontsize=20)

# Show plot
plt.show()

