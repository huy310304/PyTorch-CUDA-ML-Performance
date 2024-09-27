import torch
from torch import nn
import matplotlib.pyplot as plt
import time
from torch.profiler import profile, record_function, ProfilerActivity

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create some data using the linear regression formula of y = w * X + b
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 10
step = 0.000002

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Split the data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Create a linear model by subclassing nn.Module
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
        
        # Initialize weights and bias to 0
        nn.init.zeros_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print("Model state dict:", model_1.state_dict())

# Set the model to use the target device
model_1.to(device)

# Setup loss function
loss_fn = nn.L1Loss()  # same as MAE

# Setup our optimizer
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)

# Training loop
torch.manual_seed(42)
epochs = 200

# Put data on the target device (device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Initialize lists to store loss and memory usage
train_losses = []
test_losses = []
memory_usage = []

# Start timing
start_time = time.time()

# Use PyTorch profiler to trace performance
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for epoch in range(epochs):
        model_1.train()

        # 1. Forward pass
        y_pred = model_1(X_train)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y_train)
        train_losses.append(loss.item())

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform backpropagation
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Testing
        model_1.eval()
        with torch.inference_mode():
            test_pred = model_1(X_test)
            test_loss = loss_fn(test_pred, y_test)
            test_losses.append(test_loss.item())

        # Capture memory usage
        if device == "cuda":
            memory_usage.append(torch.cuda.memory_allocated(device))

        # Print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# End timing
end_time = time.time()

# Calculate and print the training time
training_time = end_time - start_time
print(f"Training time: {training_time * 1000} miliseconds")

# Calculate throughput
throughput = len(X_train) * epochs / training_time
print(f"Throughput: {throughput:.2f} data points/second")

# Save the profiling data as a .json file for trace viewing
prof.export_chrome_trace("trace.json")

# Plotting the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss over Epochs")
plt.legend()
plt.savefig("Python_loss_plot.png")

# Plotting memory usage
if device == "cuda":
    plt.figure(figsize=(10, 7))
    plt.plot(memory_usage, label="Memory Usage (Bytes)")
    plt.xlabel("Epoch")
    plt.ylabel("Memory Usage (Bytes)")
    plt.title("Memory Usage over Epochs")
    plt.legend()
    plt.savefig("Python_memory_usage_plot.png")

# Print final model parameters
print("Final model parameters:")
print(model_1.state_dict())