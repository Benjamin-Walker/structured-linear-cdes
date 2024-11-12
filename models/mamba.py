import torch.nn as nn
from mamba_ssm import Mamba2

from models.embedding import Embedding


class MambaBlock(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(MambaBlock, self).__init__()
        self.mamba = Mamba2(d_model=input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        norm_x = self.norm(x)
        y = self.mamba(norm_x)
        y = y + x
        y = self.drop(y)
        return y


class StackedMamba(nn.Module):
    def __init__(self, num_blocks, model_dim, data_dim, label_dim, dropout_rate=0.1):
        super(StackedMamba, self).__init__()
        self.embedding = Embedding(data_dim, model_dim)
        self.blocks = nn.ModuleList(
            [MambaBlock(model_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.linear = nn.Linear(model_dim, label_dim)

    def mask_grads(self):
        pass

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.linear(x)
