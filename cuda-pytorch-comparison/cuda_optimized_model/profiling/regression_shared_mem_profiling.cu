#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

// CUDA kernel for data initialization
__global__ void initialize_data(float* X, float* y, float weight, float bias, float start, float step, int n, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curand_init(1234, i, 0, &states[i]);
        X[i] = start + i * step;
        y[i] = weight * X[i] + bias + curand_normal(&states[i]) * 0.1f; // Add some noise
    }
}

// CUDA kernel for forward pass, loss calculation, and gradient calculation using shared memory
__global__ void fused_kernel(float* X, float* y_true, float* w, float* b, float* y_pred, float* loss, float* w_grad, float* b_grad, int n) {
    __shared__ float s_X[BLOCK_SIZE];
    __shared__ float s_y_true[BLOCK_SIZE];
    __shared__ float s_loss[BLOCK_SIZE];
    __shared__ float s_w_grad[BLOCK_SIZE];
    __shared__ float s_b_grad[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (i < n) {
        s_X[tid] = X[i];
        s_y_true[tid] = y_true[i];
    }
    __syncthreads();

    float local_loss = 0.0f;
    float local_w_grad = 0.0f;
    float local_b_grad = 0.0f;

    if (i < n) {
        // Forward pass
        float pred = w[0] * s_X[tid] + b[0];
        y_pred[i] = pred;

        // Calculate the difference between prediction and true value
        float diff = pred - s_y_true[tid];

        // Compute local loss and gradients
        local_loss = diff * diff / (2 * n);  // Mean Squared Error
        local_w_grad = diff * s_X[tid] / n;
        local_b_grad = diff / n;
    }

    // Store local results in shared memory
    s_loss[tid] = local_loss;
    s_w_grad[tid] = local_w_grad;
    s_b_grad[tid] = local_b_grad;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_loss[tid] += s_loss[tid + stride];
            s_w_grad[tid] += s_w_grad[tid + stride];
            s_b_grad[tid] += s_b_grad[tid + stride];
        }
        __syncthreads();
    }

    // Write the block results to global memory
    if (tid == 0) {
        atomicAdd(loss, s_loss[0]);
        atomicAdd(w_grad, s_w_grad[0]);
        atomicAdd(b_grad, s_b_grad[0]);
    }
}

// CUDA kernel for updating weights
__global__ void update_weights(float* w, float* w_grad, float* b, float* b_grad, float lr) {
    w[0] -= lr * w_grad[0];
    b[0] -= lr * b_grad[0];
}

int main() {
    cudaSetDevice(0);

    // Timing events
    cudaEvent_t start_time, stop, epoch_start, epoch_stop;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop);
    cudaEventCreate(&epoch_start);
    cudaEventCreate(&epoch_stop);

    float forward_time = 0, loss_time = 0, grad_time = 0, update_time = 0, total_time = 0;

    // Data parameters
    const float weight = 0.7f;
    const float bias = 0.3f;
    const float start = 0.0f;
    const float end = 10.0f;
    const float step = 0.000002f;

    const int n = static_cast<int>((end - start) / step);
    const int train_split = static_cast<int>(0.8 * n);
    const int test_split = n - train_split;

    // Print out the number of data points
    std::cout << "Total number of datapoints: " << n << std::endl;
    std::cout << "Number of training datapoints: " << train_split << std::endl;
    std::cout << "Number of testing datapoints: " << test_split << std::endl;

    // Create CUDA streams
    const int numberOfStreams = 4;
    cudaStream_t streams[numberOfStreams];
    for (int i = 0; i < numberOfStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate memory on the device
    float *d_X, *d_y, *d_w, *d_b, *d_y_pred, *d_loss, *d_w_grad, *d_b_grad;
    curandState *d_states;
    cudaMalloc(&d_X, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_y_pred, n * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_w_grad, sizeof(float));
    cudaMalloc(&d_b_grad, sizeof(float));
    cudaMalloc(&d_states, n * sizeof(curandState));

    cudaEventRecord(start_time);

    // Initialize data on GPU
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initialize_data<<<blocks, BLOCK_SIZE>>>(d_X, d_y, weight, bias, start, step, n, d_states);

    // Initialize weights
    float h_w = 0.83f;
    float h_b = 0.7645f;
    cudaMemcpyAsync(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice, streams[0]);

    // Training loop
    int epochs = 200;
    float lr = 0.01f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        cudaEventRecord(epoch_start);

        // Zero the gradients and loss
        float zero = 0.0f;
        cudaMemcpyAsync(d_loss, &zero, sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        cudaMemcpyAsync(d_w_grad, &zero, sizeof(float), cudaMemcpyHostToDevice, streams[1]);
        cudaMemcpyAsync(d_b_grad, &zero, sizeof(float), cudaMemcpyHostToDevice, streams[2]);

        // Fused kernel for training data (using multiple streams)
        cudaEvent_t fused_start, fused_stop;
        cudaEventCreate(&fused_start);
        cudaEventCreate(&fused_stop);
        cudaEventRecord(fused_start);

        int segmentSize = train_split / numberOfStreams;
        for (int i = 0; i < numberOfStreams; ++i) {
            int offset = i * segmentSize;
            fused_kernel<<<(segmentSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, streams[i]>>>(
                &d_X[offset], &d_y[offset], d_w, d_b, 
                &d_y_pred[offset], d_loss, d_w_grad, d_b_grad, segmentSize);
        }

        cudaEventRecord(fused_stop);
        cudaEventSynchronize(fused_stop);
        float fused_time;
        cudaEventElapsedTime(&fused_time, fused_start, fused_stop);
        forward_time += fused_time;
        loss_time += fused_time;
        grad_time += fused_time;

        // Update weights
        cudaEvent_t update_start, update_stop;
        cudaEventCreate(&update_start);
        cudaEventCreate(&update_stop);
        cudaEventRecord(update_start);

        update_weights<<<1, 1, 0, streams[0]>>>(d_w, d_w_grad, d_b, d_b_grad, lr);

        cudaEventRecord(update_stop);
        cudaEventSynchronize(update_stop);
        float update_time_epoch;
        cudaEventElapsedTime(&update_time_epoch, update_start, update_stop);
        update_time += update_time_epoch;

        cudaEventRecord(epoch_stop);
        cudaEventSynchronize(epoch_stop);
        float epoch_time;
        cudaEventElapsedTime(&epoch_time, epoch_start, epoch_stop);
        total_time += epoch_time;

        // Print out what's happening every 10 epochs
        if (epoch % 10 == 0) {
            float h_loss, h_w, h_b;
            cudaMemcpyAsync(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
            cudaMemcpyAsync(&h_w, d_w, sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
            cudaMemcpyAsync(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost, streams[2]);
            cudaDeviceSynchronize();
            std::cout << "Epoch: " << epoch << " | Loss: " << h_loss << " | w: " << h_w << " | b: " << h_b 
                      << " | Epoch time: " << epoch_time << " ms" << std::endl;
        }

        cudaEventDestroy(fused_start);
        cudaEventDestroy(fused_stop);
        cudaEventDestroy(update_start);
        cudaEventDestroy(update_stop);
    }

    // Compute test loss
    float h_test_loss;
    cudaMemsetAsync(d_loss, 0, sizeof(float), streams[0]);
    fused_kernel<<<(test_split + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, streams[0]>>>(
        &d_X[train_split], &d_y[train_split], d_w, d_b, &d_y_pred[train_split], d_loss, d_w_grad, d_b_grad, test_split);
    cudaMemcpyAsync(&h_test_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost, streams[0]);

    // Copy final weights and bias back to host
    float h_w_final, h_b_final;
    cudaMemcpyAsync(&h_w_final, d_w, sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(&h_b_final, d_b, sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_time, stop);

    std::cout << "Final weight: " << h_w_final << " (True: " << weight << ")" << std::endl;
    std::cout << "Final bias: " << h_b_final << " (True: " << bias << ")" << std::endl;
    std::cout << "Test loss: " << h_test_loss << std::endl;
    std::cout << "Training time: " << milliseconds << " ms" << std::endl;

    std::cout << "Average time per epoch:" << std::endl;
    std::cout << "Forward pass + Loss calculation + Gradient calculation: " << forward_time / epochs << " ms" << std::endl;
    std::cout << "Weight update: " << update_time / epochs << " ms" << std::endl;
    std::cout << "Total: " << total_time / epochs << " ms" << std::endl;

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_y_pred);
    cudaFree(d_loss);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_w_grad);
    cudaFree(d_b_grad);
    cudaFree(d_states);

    // Destroy streams
    for (int i = 0; i < numberOfStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // Destroy events
    cudaEventDestroy(start_time);
    cudaEventDestroy(stop);
    cudaEventDestroy(epoch_start);
    cudaEventDestroy(epoch_stop);

    return 0;
}
