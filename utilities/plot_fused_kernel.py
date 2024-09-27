import matplotlib.pyplot as plt

# Data setup for the two bars, combining times for each version
times_fused_combined = [11.455590, 4.729890]  # Fused kernel and update_weights
times_non_fused_combined = [
    12.710655 + 12.214040 + 6.654393,  # Sum of Calculate Loss, Forward, Gradients
    6.042793                           # Update Weights
]

# Create the plot
plt.figure(figsize=(10, 2.5))

# Plot for the non-fused kernel version
plt.barh('Without Fused Kernel', times_non_fused_combined[0], color='red', height=0.3, label='Forward + Loss + Gradient')
plt.barh('Without Fused Kernel', times_non_fused_combined[1], color='blue', left=times_non_fused_combined[0], height=0.3, label='Update Weights')

# Plot for the fused kernel version
plt.barh('With Fused Kernel', times_fused_combined[0], color='red', height=0.3)
plt.barh('With Fused Kernel', times_fused_combined[1], color='blue', left=times_fused_combined[0], height=0.3)

# Adjusting labels and ticks font size
plt.xlabel('Total Time (ms)', fontsize=16)
plt.title('Kernel Execution Time: Fused vs Non-Fused with 500 Datapoints and 2000 Epochs', fontsize=16)

# Adding the time labels
plt.text(times_fused_combined[0] / 2, 1, f'{times_fused_combined[0]:.2f} ms', ha='center', va='center', fontsize=15, color='white')
plt.text(times_fused_combined[0] + times_fused_combined[1] / 2, 1, f'{times_fused_combined[1]:.2f} ms', ha='center', va='center', fontsize=15, color='white')
plt.text(times_non_fused_combined[0] / 2, 0, f'{times_non_fused_combined[0]:.2f} ms', ha='center', va='center', fontsize=15, color='white')
plt.text(times_non_fused_combined[0] + times_non_fused_combined[1] / 2, 0, f'{times_non_fused_combined[1]:.2f} ms', ha='center', va='center', fontsize=15, color='white')

# Adjust the legend
plt.legend(loc='upper right', fontsize=13)

# Remove y-axis labels and ticks
plt.yticks([0, 1], ['Without Fused\nKernel', 'With Fused\nKernel'], fontsize=20)
plt.gca().tick_params(axis='y', which='both', length=0)

# Adjust x-axis tick font size
plt.xticks(fontsize=20)

plt.tight_layout()
plt.show()
