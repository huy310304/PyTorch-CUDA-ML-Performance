# Optimizing Machine Learning with CUDA: A Comparative Study with PyTorch 

## 📝 [A brief summarization poster](https://drive.google.com/file/d/1DgrHPqi_ic-62I6c6JHDk_6YCY23tq-9/view?usp=sharing)

## Research Focus

This project focuses on optimizing machine learning models using CUDA, comparing the performance of models developed with PyTorch and those generated using Large Language Models (LLMs) like ChatGPT 4.0. The primary goal is to evaluate the optimization potential of LLM-generated CUDA code, identifying key areas where manual intervention remains necessary and proposing strategies for improvement. Additionally, the research aims to provide insights and guidance on selecting appropriate frameworks for different ML tasks based on performance, ease of use, and optimization capabilities.

## Recent Work and Updates (Fall 2024)
- Develop an iterative optimization system where LLMs propose optimizations based on `kernel performance profiling` (Nsight reports) and `prompting`, and measure the extent to which this outperforms manual intervention.
- Explored optimization techniques for minimizing synchronization overhead, applying these methods to compare performance between PyTorch and CUDA models.
- Integrated GPU-accelerated libraries, such as cuML, cuDF, and TensorRT from the RAPIDS suite as well as MLCommon Benchmarks to extend the comparison with PyTorch, focusing on performance gains in various machine learning tasks.

## Results & Conclusions

- **Limitations of LLM-Generated Code**: LLMs like ChatGPT 4.0 can generate CUDA code but often require manual optimization to reach efficiency, particularly in handling memory and kernel execution in complex tasks.

- **Optimizing LLM-Generated CUDA Models**: Achieved a 90% improvement in training time through advanced techniques like concurrent CUDA streams, fused kernels, and shared memory, using insights from NVIDIA Nsight Systems and Compute to guide optimization.

- **Iterative LLM-Based Optimization**: Developed a system using LLM-based prompting to refine CUDA code, resulting in a 50% boost in execution time and highlighting areas where LLM-generated code needs further tuning, especially in memory management and kernel efficiency.

- **Performance Gains Over PyTorch**: Optimized CUDA models reduced training time by 60% and individual operations by 80% compared to PyTorch when processing large datasets (50M+ datapoints) on NVIDIA H100/A100 GPUs.

- **CUDA vs. PyTorch Insights**: CUDA is ideal for performance-critical applications, offering substantial speed gains through low-level optimizations. In contrast, PyTorch excels in ease of use and rapid prototyping for general ML tasks.


***

## General Notes:
- Ensure the correct CUDA toolkit is loaded using: `module load cuda-toolkit-11.7.0`
- To compile CUDA C++ programs (.cu), use: `nvcc vector_add.cu -o out`
- To run the executable: `./out` 
- Profile with nvprof (Deprecated): `nvprof ./out`
- To profile with Nsight system: `nsys profile --stats=true -o out-report ./out` and the output report will be saved as `out-report.qdrep` and can be analyzed using the Nsight Systems UI or command-line tools.

## Tools and Methods

- **Programming Languages**: Python, NVIDIA CUDA
- **Machine Learning Framework**: PyTorch and CUDA ML models from scratch
- **Large Language Models (LLMs)**: ChatGPT4.0 (for generating CUDA code)
- **Profiling Tools**: 
  - *NVIDIA Nsight Systems*: For comprehensive profiling and system-level analysis of CUDA applications.
  - *NVIDIA Nsight Compute*: For detailed kernel profiling and optimization analysis in CUDA.
  - *Perfetto and PyTorch Profiler*: For profiling and analyzing PyTorch models, focusing on performance metrics such as execution time and memory usage.
- **Hardware**: 
  - *NVIDIA RTX 2080 Ti GPUs*: Utilized for running CUDA and PyTorch models.
  - *UVA GPU servers*: The primary computing resource for running and profiling experiments.
- **Visualization**:
  - *Matplotlib*: Used for creating performance comparison charts and visualizations to clearly display the results.

## [Phase 1: Learning CUDA and Optimization Techniques](./vector-add-optimization)

### Objective & Outcome:
Implement simple CUDA applications like vector addition to understand CUDA and optimize their performance. This phase demonstrated significant performance gains through GPU acceleration, laying the groundwork for scaling up to ML models.

### Progress:
- **Vector Addition Example**: Implement vector addition in both Python and CUDA to compare performance. [View results here](./simple-vector-add)
  
- **Optimization and Visualization**: Apply basic optimizations to enhance CUDA performance and Profile CUDA implementation with Nsight tools and visualized the impact of those optimizations.

## [Phase 2: Optimizing LLM-Generated CUDA Code](./cuda-ml-optimized/)

### Objective & Outcome:
Enhance LLM-generated CUDA ML code and compare with manual optimizations. This phase identified inefficiencies in LLM-generated code and showed that, with optimization, it can approach manually optimized CUDA implementations.

### Progress:
- **LLM Code Generation and Optimization**: Generate CUDA code for simple ML tasks using ChatGPT, assess initial performance, and identify inefficiencies in LLM-generated code and applied optimizations learned from Phase 1.

- **Performance Profiling**: Profile the optimized code and document improvements in execution time, memory usage, accuracy, and GPU utilization.

## [Phase 3: Performance Comparison for PyTorch and CUDA Simple ML Models](./cuda-pytorch-comparison/)

### Objective & Outcome:
Compare performance of simple ML models implemented in PyTorch and CUDA. CUDA was found to excel in performance due to GPU-specific optimizations, while PyTorch offered ease of use and flexibility.

### Progress:
- **Model Implementation**: Developed basic ML models in both PyTorch and used existing optimized ML model CUDA in Phase 2.
- **Performance Profiling**: Profiled execution time, memory usage, and GPU utilization for both implementations.
- **Benchmarking and Visualization**: Conducted benchmarking and generated visual comparisons to highlight strengths and weaknesses of each framework.

## Future Work

- **Scalability**: Extend the comparison to more complex models and larger datasets.
- **Explore Benchmark**: Implement more advanced tasks like image classification and GANs to broaden the scope of comparison.
- **Enhance CUDA Techniques**: Apply more advanced CUDA optimization strategies for better performance.
- **Advanced LLM Utilization**: Investigate how to better guide LLMs in generating highly optimized CUDA code.
- **Extended Comparisons**: Consider comparing CUDA models with other machine learning frameworks like TensorFlow.

## Acknowledgments

- Special thanks to Professor Adwait Jog for his invaluable guidance and William Kaiser for his gpu-profiling materials.

- Acknowledgment to the Department of Computer Science at the University of Virginia for providing the necessary resources and GPU servers.
