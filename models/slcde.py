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


class LinearCDE(nn.Module):
    def __init__(self, hidden_dim, data_dim, sparsity=1.0, init_std=1.0):
        super(LinearCDE, self).__init__()
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        # Define linear layers
        self.init_layer = nn.Linear(data_dim, hidden_dim, bias=True)
        self.vf_A = nn.Linear(hidden_dim, hidden_dim * (data_dim + 1), bias=False)
        self.vf_B = nn.Linear(data_dim + 1, hidden_dim, bias=False)
        # Apply custom weight initialization
        nn.init.normal_(self.init_layer.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.init_layer.bias, mean=0.0, std=init_std)
        nn.init.normal_(self.vf_A.weight, mean=0.0, std=init_std / (hidden_dim**0.5))
        nn.init.normal_(self.vf_B.weight, mean=0.0, std=init_std)

        self.register_buffer("mask", self._sparse_mask(sparsity))

        with torch.no_grad():
            self.vf_A.weight *= self.mask

    def _sparse_mask(self, sparsity):
        mask = (
            torch.rand((self.data_dim + 1) * self.hidden_dim, self.hidden_dim)
            < sparsity
        )
        return mask

    def mask_grads(self):
        self.vf_A.weight.grad *= self.mask

    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        # Add increments of time as first channel
        ts = torch.full((batch_size, seq_len, 1), 1 / (seq_len - 1), device=X.device)
        inp = torch.cat((ts, X), dim=-1)
        # Initialise the hidden state
        x0 = X[:, 0, :]
        y0 = self.init_layer(x0)
        ys = torch.zeros(batch_size, seq_len, self.hidden_dim, device=X.device)
        ys[:, 0] = y0
        y = y0

        # Recurrently calculate the hidden states
        for i in range(1, X.shape[1]):
            A = self.vf_A(y).view(-1, self.hidden_dim, self.data_dim + 1)
            y = y + torch.einsum("bij,bj->bi", A, inp[:, i]) + self.vf_B(inp[:, i])
            ys[:, i] = y

        return ys


class A5LinearCDE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        data_dim,
        label_dim,
        init_std=1.0,
        sparsity=1.0,
        dropout_rate=0.1,
    ):
        super(A5LinearCDE, self).__init__()
        # Define components: embedding, CDE, normalization, dropout, linear
        self.embedding = Embedding(label_dim, data_dim)
        self.LCDE = LinearCDE(
            hidden_dim, data_dim, init_std=init_std, sparsity=sparsity
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(hidden_dim, label_dim)
        self.label_dim = label_dim

    def mask_grads(self):
        self.LCDE.mask_grads()

    def forward(self, X):
        # Apply embedding, CDE, normalization, dropout, and final linear layer
        X = self.embedding(X)
        ys = self.LCDE(X)
        ys = self.norm(ys)
        ys = self.drop(ys)
        return self.linear(ys)
