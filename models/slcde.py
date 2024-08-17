import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        return self.weights[x]


class LinearCDE(nn.Module):
    def __init__(self, hidden_dim, data_dim, omega_dim, xi_dim, label_dim):
        super(LinearCDE, self).__init__()
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.omega_dim = omega_dim
        self.xi_dim = xi_dim

        self.init_matrix = nn.Parameter(torch.randn(hidden_dim, data_dim))
        self.init_bias = nn.Parameter(torch.randn(hidden_dim))

        self.vf_A = nn.Linear(hidden_dim, hidden_dim * omega_dim)
        self.vf_B = nn.Parameter(torch.randn(hidden_dim, xi_dim))

    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        ts = torch.linspace(0, 1, seq_len, device=X.device)
        inp = torch.cat(
            (ts.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1), X), dim=-1
        )
        x0 = X[:, 0, :]
        control_path = torch.cat((inp, inp), dim=-1)
        y0 = (
            torch.matmul(self.init_matrix, x0.t()).t() + self.init_bias
        )  # Batch matrix multiplication

        def func(y):
            vf_A = self.vf_A(y)  # Resulting shape: (batch_size, hidden_dim * omega_dim)
            vf_A = vf_A.view(-1, self.hidden_dim, self.omega_dim)
            return torch.cat(
                (vf_A, self.vf_B.unsqueeze(0).repeat(batch_size, 1, 1)), dim=-1
            )

        ys = [y0]
        y = y0
        for i in range(1, X.shape[1]):
            vec = func(y)
            y = y + torch.einsum("bij,bj->bi", vec, control_path[:, i, :]) * (
                1.0 / seq_len
            )
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        return ys


class A5LinearCDE(nn.Module):
    def __init__(self, hidden_dim, data_dim, omega_dim, xi_dim, label_dim):
        super(A5LinearCDE, self).__init__()
        self.embedding = Embedding(label_dim, data_dim)
        self.LCDE = LinearCDE(hidden_dim, data_dim, omega_dim, xi_dim, label_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, label_dim)
        self.label_dim = label_dim

    def forward(self, X):
        X = torch.stack([self.embedding(x) for x in X])
        ys = self.LCDE(X)
        ys = torch.stack([self.norm(y) for y in ys])
        ys = self.drop(ys)
        return torch.stack([self.linear(y) for y in ys])
