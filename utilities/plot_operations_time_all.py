import matplotlib.pyplot as plt
import numpy as np

# Data for comparison of PyTorch and CUDA (not fused)
labels_combined = ['Forward Pass', 'Loss Calculation', 'Backpropagation \n and Gradient Descent', 'Total Time']
pytorch_combined_times = [0.167, 0.898, 0.887 + 0.145, 2.097]  # Add total time for PyTorch
cuda_not_fused_combined_times = [0.0923693, 0.169064, 0.288108 + 0.00417552, 0.966627]  # Add total time for CUDA Not Fused
cuda_fused_time = [0, 0, 0, 0.202649]  # Replace None with 0 for CUDA Fused in the first three operations

x_combined = np.arange(len(labels_combined))  # label locations
width = 0.4  # width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plots for PyTorch, CUDA Not Fused, and CUDA Fused
bar1 = ax.bar(x_combined - width, pytorch_combined_times, width, label='PyTorch GPU', color='orange', edgecolor='black')
bar2 = ax.bar(x_combined, cuda_not_fused_combined_times, width, label='CUDA Not Fused', color='yellow', edgecolor='black')
bar3 = ax.bar(x_combined + width, cuda_fused_time, width, label='CUDA Fused', color='green', edgecolor='black')

# Labels and title
ax.set_xlabel('Operations', fontsize=16)
ax.set_ylabel('Time (Milliseconds)', fontsize=16)
ax.set_title('Comparison of PyTorch and CUDA Operation & Total Epoch Times', fontsize=18)

# X-axis tick labels and Y-axis tick labels with increased fontsize
ax.set_xticks(x_combined)
ax.set_xticklabels(labels_combined, fontsize=14)
ax.tick_params(axis='y', labelsize=14)

# Add grid and legend with bigger fonts
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=14)

# Adding labels inside bars for PyTorch
for i in range(len(bar1)):
    if pytorch_combined_times[i] is not None:
        ax.text(bar1[i].get_x() + bar1[i].get_width()/2, bar1[i].get_height()/2, 
                f'{pytorch_combined_times[i]:.3f} ms', ha='center', va='center', color='black', fontsize=12)

# Adding labels inside bars for CUDA Not Fused
for i in range(len(bar2)):
    if cuda_not_fused_combined_times[i] is not None:
        ax.text(bar2[i].get_x() + bar2[i].get_width()/2, bar2[i].get_height()/2, 
                f'{cuda_not_fused_combined_times[i]:.3f} ms', ha='center', va='center', color='black', fontsize=12)

# Adding label inside bar for CUDA Fused (only for total time)
if cuda_fused_time[-1] is not None:
    ax.text(bar3[-1].get_x() + bar3[-1].get_width()/2, bar3[-1].get_height()/2, 
            f'{cuda_fused_time[-1]:.3f} ms', ha='center', va='center', color='black', fontsize=12)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
