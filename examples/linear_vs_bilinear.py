import time

import torch
import torch.nn as nn

# Define the dimensions
batch_size = 64
seq_len = 100
hidden_dim = 128
data_dim = 64

# Set up the inputs
X = torch.randn(batch_size, seq_len, data_dim).cuda()
y = torch.randn(batch_size, hidden_dim).cuda()

# Define the layers
linear_layer = nn.Linear(hidden_dim, hidden_dim * (data_dim + 1), bias=False).cuda()
bilinear_layer = nn.Bilinear(hidden_dim, data_dim + 1, hidden_dim, bias=False).cuda()

# Initialize the layers with the same weights for comparison
with torch.no_grad():
    # Flatten the bilinear weights and copy them to the linear layer
    bilinear_weight_flat = bilinear_layer.weight.permute(0, 2, 1).reshape(
        -1, hidden_dim
    )
    linear_layer.weight.copy_(bilinear_weight_flat)

# Create a time tensor to concatenate with X
ts = torch.full((batch_size, seq_len, 1), 1 / (seq_len - 1)).cuda()
inp = torch.cat((ts, X), dim=-1)

# Compute the outputs for comparison
bilinear_output = bilinear_layer(y, inp[:, 0])
A = linear_layer(y).view(batch_size, hidden_dim, data_dim + 1)
linear_einsum_output = torch.einsum("bod,bd->bo", A, inp[:, 0])

# Check if the outputs are the same
outputs_are_close = torch.allclose(bilinear_output, linear_einsum_output, atol=1e-5)
print(f"Outputs are the same: {outputs_are_close}")

# Timing the Bilinear layer approach
start_time = time.time()
for _ in range(1000):
    _ = bilinear_layer(y, inp[:, 0])
torch.cuda.synchronize()  # Wait for GPU to finish
bilinear_time = time.time() - start_time

# Timing the Linear layer + einsum approach
start_time = time.time()
for _ in range(1000):
    A = linear_layer(y).view(batch_size, hidden_dim, data_dim + 1)
    _ = torch.einsum("bod,bd->bo", A, inp[:, 0])
torch.cuda.synchronize()  # Wait for GPU to finish
linear_einsum_time = time.time() - start_time

# Print the results
print(f"Bilinear layer time: {bilinear_time:.6f} seconds")
print(f"Linear layer + einsum time: {linear_einsum_time:.6f} seconds")
print(f"Speedup: {bilinear_time / linear_einsum_time:.2f}x")
