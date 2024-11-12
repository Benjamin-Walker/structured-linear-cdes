import torch
import torch.nn as nn

from models.embedding import Embedding


class LinearCDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sparsity=1.0, init_std=1.0):
        super(LinearCDE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Define linear layers
        self.init_layer = nn.Linear(input_dim, hidden_dim, bias=True)
        self.vf_A = nn.Linear(input_dim + 1, hidden_dim * hidden_dim, bias=False)
        self.vf_B = nn.Linear(input_dim + 1, hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
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
            torch.rand(self.hidden_dim * self.hidden_dim, self.input_dim + 1) < sparsity
        )
        return mask

    def _diag_mask(self):
        mask = torch.eye(self.hidden_dim, self.hidden_dim).view(-1)
        mask = mask.repeat(self.input_dim + 1, 1).T
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

        Bs = self.vf_B(inp[:, 1:])

        # Recurrently calculate the hidden states
        for i in range(1, X.shape[1]):
            A = self.vf_A(inp[:, i]).view(-1, self.hidden_dim, self.hidden_dim)
            y = y + torch.einsum("bij,bj->bi", A, y) + Bs[:, i - 1]
            ys[:, i] = y

        return self.linear(ys)


class LinearCDEBlock(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, init_std=1.0, sparsity=1.0, dropout_rate=0.1
    ):
        super(LinearCDEBlock, self).__init__()
        self.LCDE = LinearCDE(
            input_dim, hidden_dim, input_dim, init_std=init_std, sparsity=sparsity
        )
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(p=dropout_rate)

    def mask_grads(self):
        self.LCDE.mask_grads()

    def forward(self, X):
        # Apply LinearCDE and add skip connection
        norm_X = self.norm(X)
        ys = self.LCDE(norm_X)
        ys = ys + X  # Skip connection
        ys = self.drop(ys)  # Dropout
        return ys


class StackedLCDE(nn.Module):
    def __init__(
        self,
        num_blocks,
        hidden_dim,
        data_dim,
        embedding_dim,
        label_dim,
        init_std=1.0,
        sparsity=1.0,
        dropout_rate=0.1,
    ):
        super(StackedLCDE, self).__init__()
        self.embedding = Embedding(data_dim, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                LinearCDEBlock(
                    embedding_dim, hidden_dim, init_std, sparsity, dropout_rate
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear = nn.Linear(embedding_dim, label_dim)

    def mask_grads(self):
        for block in self.blocks:
            block.mask_grads()

    def forward(self, X):
        # Apply embedding layer
        X = self.embedding(X)
        # Pass through each block
        for block in self.blocks:
            X = block(X)
        # Final linear layer
        return self.linear(X)
