#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>

// CUDA kernel for forward pass and gradient calculation
__global__ void forward_and_gradient(float* X, float* y_true, float* w, float* b, float* y_pred, float* w_grad, float* b_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y_pred[i] = w[0] * X[i] + b[0];
        float diff = y_pred[i] - y_true[i];
        atomicAdd(w_grad, diff * X[i] / n);
        atomicAdd(b_grad, diff / n);
    }
}

// CUDA kernel for calculating the loss
__global__ void calculate_loss(float* y_pred, float* y_true, float* loss, int n) {
    __shared__ float shared_loss[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    shared_loss[tid] = 0.0f;
    if (i < n) {
        shared_loss[tid] = fabs(y_pred[i] - y_true[i]) / n;
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }
    
    // Only thread 0 adds the result to global memory
    if (tid == 0) {
        atomicAdd(loss, shared_loss[0]);
    }
}

// CUDA kernel for updating weights
__global__ void update_weights(float* w, float* w_grad, float* b, float* b_grad, float lr) {
    w[0] -= lr * w_grad[0];
    b[0] -= lr * b_grad[0];
}

int main() {
    cudaSetDevice(0);

    cudaEvent_t start_time, stop;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop);

    // Create data
    float weight = 0.7f;
    float bias = 0.3f;
    float start = 0.0f;
    float end = 10.0f;
    float step = 0.00002f;

    int n = static_cast<int>((end - start) / step);
    std::vector<float> h_X(n), h_y(n);

    for (int i = 0; i < n; ++i) {
        h_X[i] = start + i * step;
        h_y[i] = weight * h_X[i] + bias;
    }

    int train_split = static_cast<int>(0.8 * n);
    int test_split = n - train_split;

    // Allocate unified memory
    float *d_X_train, *d_y_train, *d_X_test, *d_y_test, *d_y_pred_train, *d_y_pred_test, *d_loss_train, *d_loss_test, *d_w, *d_b, *d_w_grad, *d_b_grad;
    cudaMallocManaged(&d_X_train, train_split * sizeof(float));
    cudaMallocManaged(&d_y_train, train_split * sizeof(float));
    cudaMallocManaged(&d_X_test, test_split * sizeof(float));
    cudaMallocManaged(&d_y_test, test_split * sizeof(float));
    cudaMallocManaged(&d_y_pred_train, train_split * sizeof(float));
    cudaMallocManaged(&d_y_pred_test, test_split * sizeof(float));
    cudaMallocManaged(&d_loss_train, sizeof(float));
    cudaMallocManaged(&d_loss_test, sizeof(float));
    cudaMallocManaged(&d_w, sizeof(float));
    cudaMallocManaged(&d_b, sizeof(float));
    cudaMallocManaged(&d_w_grad, sizeof(float));
    cudaMallocManaged(&d_b_grad, sizeof(float));

    // Copy data to unified memory
    std::copy(h_X.begin(), h_X.begin() + train_split, d_X_train);
    std::copy(h_y.begin(), h_y.begin() + train_split, d_y_train);
    std::copy(h_X.begin() + train_split, h_X.end(), d_X_test);
    std::copy(h_y.begin() + train_split, h_y.end(), d_y_test);

    // Start timing
    cudaEventRecord(start_time);

    // Initialize weights
    *d_w = 0.7645f;
    *d_b = 0.8300f;

    // Training loop
    int epochs = 1000;
    float lr = 0.01f;

    dim3 block_size(256);
    dim3 grid_size_train((train_split + block_size.x - 1) / block_size.x);
    dim3 grid_size_test((test_split + block_size.x - 1) / block_size.x);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Zero the gradients and loss
        *d_loss_train = 0.0f;
        *d_w_grad = 0.0f;
        *d_b_grad = 0.0f;
        *d_loss_test = 0.0f;

        // Forward pass, loss calculation, and gradient calculation for training data
        forward_and_gradient<<<grid_size_train, block_size>>>(d_X_train, d_y_train, d_w, d_b, d_y_pred_train, d_w_grad, d_b_grad, train_split);
        calculate_loss<<<grid_size_train, block_size>>>(d_y_pred_train, d_y_train, d_loss_train, train_split);

        // Update weights
        update_weights<<<1, 1>>>(d_w, d_w_grad, d_b, d_b_grad, lr);

        // Forward pass and loss calculation for testing data
        forward_and_gradient<<<grid_size_test, block_size>>>(d_X_test, d_y_test, d_w, d_b, d_y_pred_test, d_w_grad, d_b_grad, test_split);
        calculate_loss<<<grid_size_test, block_size>>>(d_y_pred_test, d_y_test, d_loss_test, test_split);

        // Print progress every 10 epochs
        if (epoch % 10 == 0) {
            cudaDeviceSynchronize();
            std::cout << "Epoch: " << epoch << " | Loss: " << *d_loss_train << " | Test loss: " << *d_loss_test << std::endl;
        }
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_time, stop);

    std::cout << "Final weight: " << *d_w << std::endl;
    std::cout << "Final bias: " << *d_b << std::endl;
    std::cout << "Training time: " << milliseconds << " ms" << std::endl;

    // Free unified memory
    cudaFree(d_X_train);
    cudaFree(d_y_train);
    cudaFree(d_X_test);
    cudaFree(d_y_test);
    cudaFree(d_y_pred_train);
    cudaFree(d_y_pred_test);
    cudaFree(d_loss_train);
    cudaFree(d_loss_test);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_w_grad);
    cudaFree(d_b_grad);

    return 0;
}
