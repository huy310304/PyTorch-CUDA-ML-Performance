import torch
from torch import nn
import time

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
        # Initialize weights and bias to match the CUDA values
        with torch.no_grad():
            self.linear_layer.weight.fill_(0.83)
            self.linear_layer.bias.fill_(0.7645)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print("Model state dict:", model_1.state_dict())

# Set the model to use the target device
model_1.to(device)

# Setup loss function
loss_fn = nn.MSELoss()  # Change to MSE for consistency with CUDA

# Setup optimizer to use simple SGD (no momentum, no adaptive learning)
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

# Training loop
torch.manual_seed(42)
epochs = 200

# Put data on the target device (device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Initialize lists to store loss
train_losses = []

# Start timing
start_time = time.time()

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

    # 5. Optimizer step (simple SGD)
    optimizer.step()

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}")

# End timing
end_time = time.time()

# Calculate and print the training time
training_time = end_time - start_time
print(f"Training time: {training_time * 1000} milliseconds")

# Print final model parameters
print("Final model parameters:")
print(model_1.state_dict())