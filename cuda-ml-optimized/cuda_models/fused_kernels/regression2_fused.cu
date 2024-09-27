#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// CUDA kernel for forward pass, loss calculation, and gradient calculation
__global__ void fused_kernel(float* X, float* y_true, float* w, float* b, float* y_pred, float* loss, float* w_grad, float* b_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Forward pass
        y_pred[i] = w[0] * X[i] + b[0];

        // Calculate the difference between prediction and true value
        float diff = y_pred[i] - y_true[i];

        // Accumulate the loss
        atomicAdd(loss, fabs(diff) / n);

        // Calculate gradients
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
    cudaSetDevice(0);

    // Create CUDA events for timing
    cudaEvent_t start_time, stop;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop);

    // Create data
    float weight = 0.7;
    float bias = 0.3;
    float start = 0.0;
    float end = 10.0;
    float step = 0.0000002;

    int n = static_cast<int>((end - start) / step);
    std::vector<float> h_X(n), h_y(n);

    for (int i = 0; i < n; ++i) {
        h_X[i] = start + i * step;
        h_y[i] = weight * h_X[i] + bias;
    }

    // Split the data
    int train_split = static_cast<int>(0.8 * n);
    int test_split = n - train_split;

    std::vector<float> h_X_train(train_split), h_y_train(train_split);
    std::vector<float> h_X_test(test_split), h_y_test(test_split);

    for (int i = 0; i < train_split; ++i) {
        h_X_train[i] = h_X[i];
        h_y_train[i] = h_y[i];
    }

    for (int i = 0; i < test_split; ++i) {
        h_X_test[i] = h_X[train_split + i];
        h_y_test[i] = h_y[train_split + i];
    }

    // Allocate memory on the device
    float *d_X_train, *d_y_train, *d_X_test, *d_y_test, *d_y_pred_train, *d_y_pred_test, *d_loss_train, *d_loss_test, *d_w, *d_b, *d_w_grad, *d_b_grad;
    cudaMalloc(&d_X_train, train_split * sizeof(float));
    cudaMalloc(&d_y_train, train_split * sizeof(float));
    cudaMalloc(&d_X_test, test_split * sizeof(float));
    cudaMalloc(&d_y_test, test_split * sizeof(float));
    cudaMalloc(&d_y_pred_train, train_split * sizeof(float));
    cudaMalloc(&d_y_pred_test, test_split * sizeof(float));
    cudaMalloc(&d_loss_train, sizeof(float));
    cudaMalloc(&d_loss_test, sizeof(float));
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_w_grad, sizeof(float));
    cudaMalloc(&d_b_grad, sizeof(float));

    // Start timing
    cudaEventRecord(start_time);

    // Copy data to the device
    cudaMemcpy(d_X_train, h_X_train.data(), train_split * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_train, h_y_train.data(), train_split * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_test, h_X_test.data(), test_split * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_test, h_y_test.data(), test_split * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize weights
    float h_w = 0.0f;
    float h_b = 0.0f;
    cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);

    // Training loop
    int epochs = 200;
    float lr = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Zero the gradients and loss
        float zero = 0.0f;
        cudaMemcpy(d_loss_train, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w_grad, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_grad, &zero, sizeof(float), cudaMemcpyHostToDevice);

        // Fused kernel for training data
        fused_kernel<<<(train_split + 255) / 256, 256>>>(d_X_train, d_y_train, d_w, d_b, d_y_pred_train, d_loss_train, d_w_grad, d_b_grad, train_split);

        // Update weights
        update_weights<<<1, 1>>>(d_w, d_w_grad, d_b, d_b_grad, lr);

        // Zero the loss for testing data
        cudaMemcpy(d_loss_test, &zero, sizeof(float), cudaMemcpyHostToDevice);

        // Forward pass and loss calculation for testing data
        fused_kernel<<<(test_split + 255) / 256, 256>>>(d_X_test, d_y_test, d_w, d_b, d_y_pred_test, d_loss_test, d_w_grad, d_b_grad, test_split);

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

    // Free device memory
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

