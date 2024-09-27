# Adjusting the tick sizes and legend fonts as per the request

import matplotlib.pyplot as plt
import numpy as np

# Data for comparison of PyTorch and CUDA (not fused)
labels_combined = ['Forward Pass', 'Loss Calculation', 'Gradient + Weight Update']

# For PyTorch, combine backpropagation and optimizer step
pytorch_combined_times = [0.167, 0.898, 0.887 + 0.145]  # in milliseconds
cuda_not_fused_combined_times = [0.0923693, 0.169064, 0.288108 + 0.00417552]  # in milliseconds

x_combined = np.arange(len(labels_combined))  # label locations
width = 0.3  # width of the bars

# Create the plot with a different color for CUDA Not Fused
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plots for PyTorch and CUDA Not Fused (with combined Gradient + Weight Update) with new colors
bar1 = ax.bar(x_combined - width/2, pytorch_combined_times, width, label='PyTorch', color='orange', edgecolor='black')
bar2 = ax.bar(x_combined + width/2, cuda_not_fused_combined_times, width, label='CUDA Not Fused', color='green', edgecolor='black')

# Labels and title with bigger fonts
ax.set_xlabel('Operations', fontsize=16)
ax.set_ylabel('Time (Milliseconds)', fontsize=16)
ax.set_title('Comparison of PyTorch and CUDA (Non Fused) ML Operation Times', fontsize=18)

# X-axis tick labels with increased fontsize
ax.set_xticks(x_combined)
ax.set_xticklabels(labels_combined, fontsize=14)

# Y-axis tick labels with increased fontsize
ax.tick_params(axis='y', labelsize=20)

# Add grid and legend with bigger fonts
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=18)

# Adding labels inside bars for PyTorch
for i in range(len(bar1)):
    ax.text(bar1[i].get_x() + bar1[i].get_width()/2, bar1[i].get_height()/2, 
            f'{pytorch_combined_times[i]:.3f} ms', ha='center', va='center', color='black', fontsize=12)

# Adding labels inside bars for CUDA Not Fused
for i in range(len(bar2)):
    ax.text(bar2[i].get_x() + bar2[i].get_width()/2, bar2[i].get_height()/2, 
            f'{cuda_not_fused_combined_times[i]:.3f} ms', ha='center', va='center', color='black', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
