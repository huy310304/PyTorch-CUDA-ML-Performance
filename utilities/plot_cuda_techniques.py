import matplotlib.pyplot as plt
import numpy as np

# New data provided for the plot
data_points = np.array([500, 5000, 50_000, 500_000, 5_000_000, 50_000_000])
execution_time_default = np.array([13.2624, 19.3425, 65.0445, 382.7136, 3698.3840, 36199])
execution_time_4_stream = np.array([5.6981, 8.7815, 53.9097, 359.234, 3699, 36117])
execution_time_fused_kernels = np.array([6.06493, 10.0402, 54.2668, 265.441, 2536.74, 24209])
execution_time_non_shared = np.array([3.86458, 4.17939, 29.0099, 246.669, 1454.13, 13916.4])
execution_time_all_shared = np.array([3.8684, 3.9203, 4.3834, 9.4840, 89, 758])

# Plotting the data
plt.figure(figsize=(12, 8))  # Adjusted figure size for larger axis and numbers
plt.plot(data_points, execution_time_default, 'o-', label='Default', color='blue')
plt.plot(data_points, execution_time_4_stream, 's-', label='4 Stream Concurrent', color='green')
plt.plot(data_points, execution_time_fused_kernels, '^-', label='Fused Kernels', color='orange')
plt.plot(data_points, execution_time_non_shared, 'd-', label='All Optimizaztion - Global Memory', color='purple')
plt.plot(data_points, execution_time_all_shared, '*-', label='All Optimizaztion - Shared Memory', color='red')

# Setting log scale for both x and y-axis
plt.xscale('log')
plt.yscale('log')

# Adjusting axis labels font size
plt.xlabel('Number of Data Points', fontsize=16)
plt.ylabel('Execution Time (ms)', fontsize=16)

# Adjusting the title font size
plt.title('Execution Time Scaling with Data Size Increase', fontsize=18)

# Adjusting ticks font size
plt.xticks([500, 5000, 50_000, 500_000, 5_000_000, 50_000_000], 
           labels=['500', '5K', '50K', '500K', '5M', '50M'], fontsize=20)
plt.yticks(fontsize=20)

# Adding legend with bigger font size
plt.legend(title='Optimization Method', fontsize=14, title_fontsize=14)

# Displaying the plot
plt.show()
