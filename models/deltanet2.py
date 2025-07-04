""" "
This model uses the modified flash linear attention package from:
    https://github.com/automl/unlocking_state_tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import DeltaNet


class Block(nn.Module):
    """
    A block containing a modified DeltaNet layer which can have negative eigenvalues.
    """

    def __init__(
        self,
        model_dim: int,
        layer_idx: int = 0,
        dropout_rate: float = 0.1,
        sigmoid_scale: float = 1.0,
        use_glu: bool = False,
    ):
        super().__init__()
        # Keep track of which layer index this is
        self.layer_idx = layer_idx
        self.model_dim = model_dim

        self.layer = DeltaNet(
            hidden_size=model_dim, sigmoid_scale=sigmoid_scale
        ).bfloat16()

        self.norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(p=dropout_rate)

        self.use_glu = use_glu
        if self.use_glu:
            # For GLU, expand from model_dim to 2*model_dim
            self.post_linear = nn.Linear(model_dim, 2 * model_dim)
        else:
            self.post_linear = None

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor = None,
    ):
        out = self.layer(x.bfloat16())
        y = out[0].float()
        new_v_first = v_first

        # Apply the residual connection
        y = y + x

        # Optional GLU
        if self.use_glu:
            y_glu = self.post_linear(y)
            y_glu = F.glu(y_glu, dim=-1)
            y = y + y_glu  # second residual

        # Norm + Dropout
        y = self.norm(y)
        y = self.drop(y)

        return y, new_v_first


class StackedBlock(nn.Module):
    """
    A stack of multiple Blocks. We:
     1) do an Embedding
     2) pass (x, v_first) successively through each block
     3) final linear
    """

    def __init__(
        self,
        num_blocks: int,
        model_dim: int,
        data_dim: int,
        label_dim: int,
        dropout_rate: float = 0.1,
        sigmoid_scale: float = 1.0,
        use_glu: bool = False,
        second_embedding: bool = False,
    ):
        super().__init__()
        self.second_embedding = second_embedding
        embedding_dim = model_dim // 2 if second_embedding else model_dim

        self.embedding = nn.Embedding(data_dim, embedding_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, embedding_dim)

        # Build the blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    layer_idx=i,
                    model_dim=model_dim,
                    sigmoid_scale=sigmoid_scale,
                    dropout_rate=dropout_rate,
                    use_glu=use_glu,
                )
                for i in range(num_blocks)
            ]
        )

        # Final projection
        self.linear = nn.Linear(model_dim, label_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.second_embedding:
            x = self.embedding(x)  # [B, T, model_dim]
        else:
            x = torch.cat(
                [self.embedding(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )

        # We track v_first across blocks. Start None
        v_first = None

        for block in self.blocks:
            x, v_first = block(x, v_first=v_first)

        return self.linear(x)  # [B, T, label_dim]

    def mask_grads(self):
        pass
