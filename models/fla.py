import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import (
    DeltaNet,
    GatedDeltaNet,
    GatedDeltaProduct,
    RWKV6Attention,
    RWKV7Attention,
)


class Block(nn.Module):
    """
    A single block that can internally use one of:
      DeltaNet, GatedDeltaNet, or RWKV7Attention.
    """

    def __init__(
        self,
        model_dim: int,
        layer_idx: int = 0,
        layer_type: str = "deltanet",
        dropout_rate: float = 0.1,
        rank: int = 1,
        gated: bool = True,
        use_glu: bool = False,
    ):
        super().__init__()
        # Keep track of which layer index this is
        self.layer_idx = layer_idx
        self.layer_type = layer_type.lower()
        self.model_dim = model_dim

        layer_map = {
            "deltanet": DeltaNet,
            "gateddeltanet": GatedDeltaNet,
            "rwkv7": RWKV7Attention,
            "rwkv6": RWKV6Attention,
            "deltaproduct": GatedDeltaProduct,
        }
        if self.layer_type not in layer_map:
            raise ValueError(
                f"Unknown layer_type '{layer_type}'. "
                f"Choose from {list(layer_map.keys())}."
            )

        # Instantiate the chosen layer
        if self.layer_type == "rwkv7" or self.layer_type == "rwkv6":
            self.layer = layer_map[self.layer_type](
                mode="chunk",
                hidden_size=model_dim,
                layer_idx=layer_idx,
            ).bfloat16()
        elif self.layer_type == "deltaproduct":
            self.layer = layer_map[self.layer_type](
                hidden_size=model_dim,
                num_householder=rank,
                use_forget_gate=gated,
                allow_neg_eigval=True,
                layer_idx=layer_idx,
            ).bfloat16()
        else:
            self.layer = layer_map[self.layer_type](
                hidden_size=model_dim, layer_idx=layer_idx
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
        if self.layer_type == "rwkv7" or self.layer_type == "rwkv6":
            out = self.layer(x.bfloat16(), v_first=v_first)
            y = out[0].float()
            if self.layer_type == "rwkv7":
                new_v_first = out[3]
            elif self.layer_type == "rwkv6":
                new_v_first = out[2]
        else:
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
        layer_type: str = "deltanet",
        dropout_rate: float = 0.1,
        rank: int = 1,
        gated: bool = True,
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
                    layer_type=layer_type,
                    dropout_rate=dropout_rate,
                    use_glu=use_glu,
                    rank=rank,
                    gated=gated,
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
