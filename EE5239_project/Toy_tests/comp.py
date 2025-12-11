import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_target_matrix(m, n, true_rank, noise_level=0.1):
    U = torch.randn(m, true_rank)
    V = torch.randn(n, true_rank)
    Y = torch.matmul(U, V.t())
    noise = torch.randn(m, n) * noise_level
    return Y + noise

class MatrixFactorization(nn.Module):
    def __init__(self, m, n, rank, init_method='lora', target_matrix=None):
        super().__init__()
        self.init_method = init_method
        
        if init_method == 'lora':
            # Standard LoRA: A is random Gaussian, B is random (or zero)
            # We use small random values for both to ensure gradient flow
            self.A = nn.Parameter(torch.randn(m, rank) * 0.01)
            self.B = nn.Parameter(torch.randn(n, rank) * 0.01)
            
        elif init_method == 'pissa':
            if target_matrix is None:
                raise ValueError("PiSSA requires the target matrix for initialization")
            
            # PiSSA: Initialize using SVD of the target matrix
            # Y ~= U * S * V^T
            # We want AB^T = Y_approx
            # So we set A = U * sqrt(S) and B = V * sqrt(S)
            
            print("Computing SVD for PiSSA initialization...")
            U, S, Vh = torch.linalg.svd(target_matrix, full_matrices=False)
            
            # Truncate to desired rank
            U_r = U[:, :rank]
            S_r = S[:rank]
            V_r = Vh[:rank, :].t() # Transpose Vh to get V
            
            # Distribute the singular values (sqrt) to balance A and B
            S_sqrt = torch.diag(torch.sqrt(S_r))
            
            # Initialize parameters
            # We wrap them in nn.Parameter to make them trainable
            self.A = nn.Parameter(torch.matmul(U_r, S_sqrt))
            self.B = nn.Parameter(torch.matmul(V_r, S_sqrt))

    def forward(self):
        return torch.matmul(self.A, self.B.t())

def run_optimization(Y, rank, method='lora', steps=1000, lr=0.01):
    m, n = Y.shape
    model = MatrixFactorization(m, n, rank, init_method=method, target_matrix=Y)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for step in range(steps):
        optimizer.zero_grad()
        Y_pred = model()
        loss = torch.norm(Y - Y_pred, p='fro')**2
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    return loss_history

# --- Execution ---

# 1. Setup
M, N = 100, 100
TRUE_RANK = 20
APPROX_RANK = 10 # We are compressing the info
STEPS = 500
LR = 0.01

# 2. Generate Data
Y = generate_target_matrix(M, N, TRUE_RANK, noise_level=0.1)

# 3. Run Standard LoRA
print("Running Standard LoRA...")
losses_lora = run_optimization(Y, APPROX_RANK, method='lora', steps=STEPS, lr=LR)

# 4. Run PiSSA
print("Running PiSSA...")
losses_pissa = run_optimization(Y, APPROX_RANK, method='pissa', steps=STEPS, lr=LR)

# 5. Visualization
plt.figure(figsize=(10, 6))

# Plot LoRA
plt.plot(losses_lora, label='Standard LoRA (Random Init)', color='red', alpha=0.7)

# Plot PiSSA
plt.plot(losses_pissa, label='PiSSA (SVD Init)', color='blue', linewidth=2)

plt.yscale('log')
plt.xlabel('Optimization Steps')
plt.ylabel('Frobenius Norm Loss (Log Scale)')
plt.title(f'Initialization Impact: LoRA vs PiSSA (Rank {APPROX_RANK})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("Comp.png")