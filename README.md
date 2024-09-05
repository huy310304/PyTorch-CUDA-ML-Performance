# Optimizing Machine Learning with CUDA: A Comparative Study with PyTorch 

## 📝 [A brief summarization poster](https://drive.google.com/file/d/1-79boy7_EZHLIIbIy23dW9GVcFYegx6w/view?usp=sharing)

## Research Focus

This project aims to compare the performance of a PyTorch machine learning model and a CUDA machine learning model generated using Large Language Models (LLMs). The primary goal is to evaluate the optimization capabilities of LLMs in generating CUDA code for machine learning tasks, identify any limitations, and determine areas for improvement. Additionally, this research seeks to provide guidance on framework selection for different ML tasks based on the findings.

## Recent Work and Updates (Fall 2024)
- Explored optimization techniques for minimizing synchronization overhead, applying these methods to compare performance between PyTorch and CUDA models.
- Integrated GPU-accelerated libraries, such as cuML from the RAPIDS suite, to extend the comparison with PyTorch, focusing on performance gains in various machine learning tasks.

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

## [Phase 2: Optimizing LLM-Generated CUDA Code](./inital-LLMs-ml/)

### Objective & Outcome:
Enhance LLM-generated CUDA ML code and compare with manual optimizations. This phase identified inefficiencies in LLM-generated code and showed that, with optimization, it can approach manually optimized CUDA implementations.

### Progress:
- **LLM Code Generation and Optimization**: Generate CUDA code for simple ML tasks using ChatGPT, assess initial performance, and identify inefficiencies in LLM-generated code and applied optimizations learned from Phase 1.

- **Performance Profiling**: Profile the optimized code and document improvements in execution time, memory usage, accuracy, and GPU utilization.

## Phase 3: Performance Comparison for PyTorch and CUDA Simple ML Models

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

- Special thanks to Professor Adwait Jog for guidance and support.
- Acknowledgment to the Department of Computer Science at the University of Virginia for providing the necessary resources and GPU servers.
