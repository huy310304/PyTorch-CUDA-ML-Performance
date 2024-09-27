#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CUDA kernel to initialize the input X array and output y array
__global__ void initialize_arrays(float* X, float* y, float weight, float bias, float start, float step, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        X[i] = start + i * step;
        y[i] = weight * X[i] + bias;
    }
}

// CUDA kernel for forward pass
__global__ void forward(float* X, float* w, float* b, float* y_pred, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y_pred[i] = w[0] * X[i] + b[0];
    }
}

// CUDA kernel for calculating the loss
__global__ void calculate_loss(float* y_pred, float* y_true, float* loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(loss, fabs(y_pred[i] - y_true[i]) / n);
    }
}

// CUDA kernel for calculating gradients
__global__ void calculate_gradients(float* X, float* y_pred, float* y_true, float* w_grad, float* b_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = y_pred[i] - y_true[i];
        atomicAdd(w_grad, diff * X[i] / n);
        atomicAdd(b_grad, diff / n);
    }
}

// CUDA kernel for updating weights
__global__ void update_weights(float* w, float* w_grad, float* b, float* b_grad, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        w[0] -= lr * w_grad[0];
        b[0] -= lr * b_grad[0];
    }
}

int main() {
    // Setup device
    int deviceId;
    cudaGetDevice(&deviceId);

    // Create CUDA events for timing
    cudaEvent_t start_time, stop;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop);

    // Parameters
    float weight = 0.7;
    float bias = 0.3;
    float start = 0.0;
    float end = 10.0;
    float step = 0.0000002;

    int n = static_cast<int>((end - start) / step);

    // Split the data
    int train_split = static_cast<int>(0.8 * n);
    int test_split = n - train_split;

    // Allocate managed memory for all arrays
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

    // Start timing
    cudaEventRecord(start_time);

    // Initialize arrays on the device
    int blockSize = 256;
    int numBlocksTrain = (train_split + blockSize - 1) / blockSize;
    int numBlocksTest = (test_split + blockSize - 1) / blockSize;

    // Initialize training data
    initialize_arrays<<<numBlocksTrain, blockSize>>>(d_X_train, d_y_train, weight, bias, start, step, train_split);

    // Initialize testing data
    float test_start = start + train_split * step;
    initialize_arrays<<<numBlocksTest, blockSize>>>(d_X_test, d_y_test, weight, bias, test_start, step, test_split);

    // Prefetch data to the GPU
    cudaMemPrefetchAsync(d_X_train, train_split * sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_y_train, train_split * sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_X_test, test_split * sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_y_test, test_split * sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_y_pred_train, train_split * sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_y_pred_test, test_split * sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_loss_train, sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_loss_test, sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_w, sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_b, sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_w_grad, sizeof(float), deviceId);
    cudaMemPrefetchAsync(d_b_grad, sizeof(float), deviceId);

    // Initialize weights
    float h_w = 0.0f;
    float h_b = 0.0f;
    cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);

    // Training loop
    int epochs = 200;
    float lr = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Zero the gradients
        float zero = 0.0f;
        cudaMemcpy(d_loss_train, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w_grad, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_grad, &zero, sizeof(float), cudaMemcpyHostToDevice);

        // Forward pass for training data
        forward<<<numBlocksTrain, blockSize>>>(d_X_train, d_w, d_b, d_y_pred_train, train_split);

        // Calculate loss for training data
        calculate_loss<<<numBlocksTrain, blockSize>>>(d_y_pred_train, d_y_train, d_loss_train, train_split);

        // Calculate gradients
        calculate_gradients<<<numBlocksTrain, blockSize>>>(d_X_train, d_y_pred_train, d_y_train, d_w_grad, d_b_grad, train_split);

        // Update weights
        update_weights<<<1, 1>>>(d_w, d_w_grad, d_b, d_b_grad, lr);

        // Zero the loss for testing data
        cudaMemcpy(d_loss_test, &zero, sizeof(float), cudaMemcpyHostToDevice);

        // Forward pass for testing data
        forward<<<numBlocksTest, blockSize>>>(d_X_test, d_w, d_b, d_y_pred_test, test_split);

        // Calculate loss for testing data
        calculate_loss<<<numBlocksTest, blockSize>>>(d_y_pred_test, d_y_test, d_loss_test, test_split);

        // Print out what's happening every 10 epochs
        if (epoch % 10 == 0) {
            float h_loss_train, h_loss_test;
            cudaMemcpy(&h_loss_train, d_loss_train, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_loss_test, d_loss_test, sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Epoch: " << epoch << " | Loss: " << h_loss_train << " | Test loss: " << h_loss_test << std::endl;
        }
    }

    // Copy final weights and bias back to host
    cudaMemcpy(&h_w, d_w, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_time, stop);

    // Output the final weight and bias
    std::cout << "Final weight: " << h_w << std::endl;
    std::cout << "Final bias: " << h_b << std::endl;
    std::cout << "Training time: " << milliseconds << " ms" << std::endl;

    // Prefetch result back to CPU before checking it
    cudaMemPrefetchAsync(d_loss_train, sizeof(float), cudaCpuDeviceId);
    cudaMemPrefetchAsync(d_loss_test, sizeof(float), cudaCpuDeviceId);

    // Free managed memory
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

