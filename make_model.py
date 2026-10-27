import torch
import torch.nn as nn

class TinyGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 3 * 32 * 32)

    def forward(self, z):
        return torch.tanh(self.fc(z)).view(-1, 3, 32, 32)

G = TinyGenerator().eval()
traced = torch.jit.trace(G, torch.randn(1, 100))
traced.save("generator.pt")
print("saved generator.pt")