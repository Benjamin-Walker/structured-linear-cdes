import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        # Create embedding weights
        self.weights = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        # Retrieve embeddings by index
        return self.weights[x]
