import matplotlib.pyplot as plt
import numpy as np

# Data
init_methods = [
    "Init on CPU, MallocManaged",
    "Init on GPU, MallocManaged",
    "Init on GPU, MallocManaged and Prefetch"
]

# Kernel execution times split
add_vectors_into_time = [9.5, 0.05, 0.05]
init_with_time = [None, 5.65, 0.05]

# Memory transfer times
htod_transfer_time = [3.0, 0.0, 0.0]
dtoh_transfer_time = [0.6, 0.6, 0.6]

# Plotting the split data as two separate line graphs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Kernel Execution Time split
ax1.plot(init_methods, add_vectors_into_time, marker='o', linestyle='-', color='orange', label='AddVectorsInto Time (ms)')
ax1.plot(init_methods, init_with_time, marker='o', linestyle='-', color='green', label='InitWith Time (ms)')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Kernel Execution Time for Different Initialization Methods')
ax1.legend()
ax1.grid(True)

# Memory Transfer Times
ax2.plot(init_methods, htod_transfer_time, marker='o', linestyle='-', color='red', label='HtoD Transfer Time (ms)')
ax2.plot(init_methods, dtoh_transfer_time, marker='o', linestyle='-', color='blue', label='DtoH Transfer Time (ms)')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Memory Transfer Times for Different Initialization Methods')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
