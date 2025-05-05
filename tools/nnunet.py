import torch
from unet3d.model import UNet3D


model = UNet3D(
    in_channels=1,
    out_channels=1,
    num_layers=4,
    num_filters=[32, 64, 128, 256],
    kernel_size=3,
    padding=1,
    activation='relu',
    normalization='batch',
    dropout=0.5,
    pool_size=2,
    upsample_mode='default',
).to(device='cpu')


#Define the input tensor
# Requires a 5D tensor (batch_size, channels, depth, height, width)
input_tensor = torch.randn(1, 1, 64, 64, 64).to(device='cpu')

print("Input tensor shape:", input_tensor.shape)

#Perform a forward pass
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)

#Compute the loss
criterion = torch.nn.MSELoss()

# Define a target tensor with the same shape as the output tensor
target_tensor = torch.randn(1, 1, 64, 64, 64).to(device='cpu')

# Compute the loss
loss = criterion(output_tensor, target_tensor)
print("Loss:", loss.item())

# Perform a backward pass
loss.backward()
print("Gradients computed for the model parameters.")

#Let's define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Perform a step of optimization
optimizer.step()
print("Optimizer step performed.")


print("Training loop example:")
#Let's try to define a training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    output_tensor = model(input_tensor)
    
    # Compute the loss
    loss = criterion(output_tensor, target_tensor)
    
    # Backward pass
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
