# CUDA Vector Addition Optimization

This folder explores various optimization techniques for performing vector addition using CUDA. The vector addition involves initializing vectors, performing the addition on the GPU, checking the results on the CPU, and then freeing the allocated memory. The project focuses on how different memory management and initialization strategies affect performance, particularly in the context of CUDA's unified memory (`cudaMallocManaged`).

## Code Overview

1. **Initialization on the CPU using `cudaMallocManaged`.**
2. **Initialization on the GPU using `cudaMallocManaged`.**
3. **Initialization on the GPU using `cudaMallocManaged` combined with `cudaMemPrefetchAsync` for prefetching memory on the GPU.**

### Key Observations

- **Initialization on GPU:** 
  - Faster than initialization on the CPU because there’s no need to wait for Host-to-Device (HtoD) memory transfers.
  - Kernel execution is faster since the data is already on the GPU.
  - This method scales well with larger datasets.

- **Prefetching:**
  - Prefetching memory to the GPU using `cudaMemPrefetchAsync` results in a significant reduction in initialization kernel execution time.
  - This approach avoids waiting for data to be transferred from unified memory to the device during kernel execution, leading to much faster kernel launches.

## Performance Analysis

The performance was measured in terms of kernel execution time and memory transfer time. Below are the results based on different initialization methods:

### Kernel Execution Time

![Kernel Execution Time](./images/kernel_execution_time_comparison.png)

#### Analysis:
- **AddVectorsInto**: 
  - The time taken by the `AddVectorsInto` kernel is significantly reduced when the data is initialized on the GPU. In the CPU initialization scenario, the kernel execution takes longer because the data needs to be transferred from the CPU to the GPU before the addition can take place.
  - When initialization is done on the GPU, the execution time for `AddVectorsInto` drops sharply because the data is already on the GPU, ready for processing.

- **InitWith**: 
  - The `InitWith` kernel takes longer when the data is initialized on the GPU using `cudaMallocManaged` without prefetching. This is because, even though the data is in unified memory, it may not yet be physically present on the GPU, causing delays as data is transferred.
  - With `cudaMemPrefetchAsync`, the `InitWith` time is almost negligible because the data is already preloaded into the GPU memory, eliminating any wait time.

### Memory Transfer Times

![Memory Transfer Times](./images/memory_transfer_time_comparison.png)

#### Analysis:
- **HtoD Transfer Time**:
  - Host-to-Device transfer time is a significant factor when initializing on the CPU. Since the data needs to be moved from the host (CPU) to the device (GPU), this introduces additional overhead.
  - In the GPU-based initialization methods, the HtoD transfer time is eliminated because the data is already on the GPU, hence no transfer is necessary.

- **DtoH Transfer Time**:
  - Device-to-Host transfer time remains consistent across all methods because the data is always checked on the CPU after the computation. This operation doesn't change regardless of where the data was initially located or how it was managed.

### Using Non-default Concurrent Streams
... 

## Conclusion

The experiments demonstrate that initializing data on the GPU and prefetching memory significantly improve performance by reducing kernel execution time and eliminating unnecessary memory transfers. These optimizations are especially beneficial for larger datasets, where the overhead of memory transfers can become a bottleneck.

## Reference
- Codes was learned and used from [NVIDIA DLI Course: Getting Started with Accelerated Computing in CUDA C/C++](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-04+V1)

