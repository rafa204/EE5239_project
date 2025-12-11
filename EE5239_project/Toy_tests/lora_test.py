import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_target_matrix(m, n, true_rank, noise_level=0.1):
    """
    Generates a synthetic matrix Y = U * V^T + noise
    """
    U = torch.randn(m, true_rank)
    V = torch.randn(n, true_rank)
    Y = torch.matmul(U, V.t())
    
    # Add noise to make it realistic
    noise = torch.randn(m, n) * noise_level
    return Y + noise

class LoRAFactorization(nn.Module):
    """
    Represents the decomposition Y ~= AB^T.
    """
    def __init__(self, m, n, rank):
        super().__init__()
        # Initialize A and B
        # In LoRA papers, often A is N(0, sigma) and B is 0, but for
        # pure factorization from scratch, we initialize both randomly.
        self.A = nn.Parameter(torch.randn(m, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(n, rank) * 0.01)

    def forward(self):
        return torch.matmul(self.A, self.B.t())

def run_lora_optimization(Y, rank, steps=250, lr=0.01):
    """
    Optimizes A and B using Gradient Descent (Adam).
    """
    m, n = Y.shape
    model = LoRAFactorization(m, n, rank)
    
    # Using Adam, the standard for LoRA training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    print(f"Starting LoRA Optimization (Rank={rank})...")
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass: Compute reconstruction
        Y_pred = model()
        
        # Compute Loss: Frobenius Norm squared (equivalent to sum of squared errors)
        loss = torch.norm(Y - Y_pred, p='fro')**2
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if step % 500 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    print(f"Final Loss: {loss_history[-1]:.4f}")
    return loss_history, model.A.detach(), model.B.detach()

# --- Execution ---

# 1. Configuration
M, N = 100, 100   # Dimensions of Y
TRUE_RANK = 10    # The "actual" information content
APPROX_RANK = 10  # The rank we are forcing our model to learn

# 2. Generate Data
Y = generate_target_matrix(M, N, TRUE_RANK)

# 3. Run LoRA (Gradient Descent) Optimizer
lora_losses, A_hat, B_hat = run_lora_optimization(Y, APPROX_RANK, steps=500, lr=0.02)

print(Y)
print(A_hat @ B_hat.t())

# 4. Quick Visualization of Convergence
plt.figure(figsize=(10, 6))
plt.plot(lora_losses, label='LoRA (Adam)')
plt.yscale('log')
plt.xlabel('Optimization Steps')
plt.ylabel('Frobenius Norm Loss (Log Scale)')
plt.title('LoRA Matrix Factorization Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Lora.png")