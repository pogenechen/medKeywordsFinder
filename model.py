from torch import nn
class ContrastiveEmbedder(nn.Module):
    def __init__(self, input_dim=768, projection_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.projection(x), dim=-1)