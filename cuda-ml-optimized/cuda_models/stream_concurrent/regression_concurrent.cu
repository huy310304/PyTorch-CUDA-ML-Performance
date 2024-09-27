#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>

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

    // Allocate pinned memory on the host
    float *h_X_train, *h_y_train, *h_X_test, *h_y_test;
    cudaMallocHost(&h_X_train, train_split * sizeof(float));
    cudaMallocHost(&h_y_train, train_split * sizeof(float));
    cudaMallocHost(&h_X_test, test_split * sizeof(float));
    cudaMallocHost(&h_y_test, test_split * sizeof(float));

    // Copy data to pinned memory
    std::copy(h_X.begin(), h_X.begin() + train_split, h_X_train);
    std::copy(h_y.begin(), h_y.begin() + train_split, h_y_train);
    std::copy(h_X.begin() + train_split, h_X.end(), h_X_test);
    std::copy(h_y.begin() + train_split, h_y.end(), h_y_test);

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

    // Create CUDA streams
    const int numberOfStreams = 4;
    cudaStream_t streams[numberOfStreams];
    for (int i = 0; i < numberOfStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Asynchronously copy data to the device using streams
    cudaMemcpyAsync(d_X_train, h_X_train, train_split * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_y_train, h_y_train, train_split * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_X_test, h_X_test, test_split * sizeof(float), cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpyAsync(d_y_test, h_y_test, test_split * sizeof(float), cudaMemcpyHostToDevice, streams[1]);

    // Initialize weights
    float h_w = 0.0f;
    float h_b = 0.0f;
    cudaMemcpyAsync(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice, streams[0]);

    // Training loop
    int epochs = 200;
    float lr = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Zero the gradients
        float zero = 0.0f;
        cudaMemcpyAsync(d_loss_train, &zero, sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        cudaMemcpyAsync(d_w_grad, &zero, sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        cudaMemcpyAsync(d_b_grad, &zero, sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        cudaMemcpyAsync(d_loss_test, &zero, sizeof(float), cudaMemcpyHostToDevice, streams[1]);

        // Forward pass for training data
        forward<<<(train_split + 255) / 256, 256, 0, streams[0]>>>(d_X_train, d_w, d_b, d_y_pred_train, train_split);

        // Calculate loss for training data
        calculate_loss<<<(train_split + 255) / 256, 256, 0, streams[0]>>>(d_y_pred_train, d_y_train, d_loss_train, train_split);

        // Calculate gradients
        calculate_gradients<<<(train_split + 255) / 256, 256, 0, streams[0]>>>(d_X_train, d_y_pred_train, d_y_train, d_w_grad, d_b_grad, train_split);

        // Update weights
        update_weights<<<1, 1, 0, streams[0]>>>(d_w, d_w_grad, d_b, d_b_grad, lr);

        // Forward pass for testing data
        forward<<<(test_split + 255) / 256, 256, 0, streams[1]>>>(d_X_test, d_w, d_b, d_y_pred_test, test_split);

        // Calculate loss for testing data
        calculate_loss<<<(test_split + 255) / 256, 256, 0, streams[1]>>>(d_y_pred_test, d_y_test, d_loss_test, test_split);

        // Print out what's happening every 10 epochs
        if (epoch % 10 == 0) {
            cudaDeviceSynchronize(); // Ensure all operations are complete before reading results
            float h_loss_train, h_loss_test;
            cudaMemcpyAsync(&h_loss_train, d_loss_train, sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
            cudaMemcpyAsync(&h_loss_test, d_loss_test, sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
            cudaDeviceSynchronize(); // Ensure memory copies are complete before printing
            std::cout << "Epoch: " << epoch << " | Loss: " << h_loss_train << " | Test loss: " << h_loss_test << std::endl;
        }
    }

    // Copy final weights and bias back to host
    cudaMemcpyAsync(&h_w, d_w, sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
    cudaDeviceSynchronize(); // Ensure final copies are complete

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

    // Free pinned host memory
    cudaFreeHost(h_X_train);
    cudaFreeHost(h_y_train);
    cudaFreeHost(h_X_test);
    cudaFreeHost(h_y_test);

    // Destroy streams
    for (int i = 0; i < numberOfStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
