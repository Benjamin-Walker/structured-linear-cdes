import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from mamba_ssm import Mamba2


class MambaBlock(nn.Module):
    """
    A single Mamba block that applies:
      1. Mamba2 module
      2. (Optionally) a linear layer + GLU activation,
      3. Residual connection
      4. Layer Normalization
      5. Dropout

    Args:
        model_dim (int): Dimensionality of the model (d_model).
        dropout_rate (float): Probability of an element to be zeroed in Dropout.
        use_glu (bool): Whether to apply a Linear -> GLU stage after the residual.
    """

    def __init__(
        self, model_dim: int, dropout_rate: float = 0.1, use_glu: bool = False
    ):
        super().__init__()
        self.mamba = Mamba2(d_model=model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(p=dropout_rate)

        self.use_glu = use_glu
        if self.use_glu:
            # The linear expands from model_dim to 2*model_dim
            # so that GLU can split it into two halves of model_dim each
            self.post_linear = nn.Linear(model_dim, 2 * model_dim)
        else:
            self.post_linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MambaBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape (batch_size, seq_len, model_dim).
        """

        # Mamba2 module
        y = self.mamba(x)
        y = y + x

        # Optional: Linear -> GLU
        if self.use_glu:
            # shape: (batch_size, seq_len, 2 * model_dim)
            y_glu = self.post_linear(y)
            # shape: (batch_size, seq_len, model_dim)
            y_glu = F.glu(y_glu, dim=-1)
            y = y + y_glu

        # Layer Normalization
        y = self.norm(y)

        # Dropout
        y = self.drop(y)

        return y


class StackedMamba(nn.Module):
    """
    A stack of multiple MambaBlocks, preceded by an embedding layer
    and followed by a linear projection.

    Args:
        num_blocks (int): Number of MambaBlocks to stack.
        model_dim (int): Dimensionality of embeddings and Mamba blocks.
        data_dim (int): Size of the vocabulary (if input is token IDs).
        label_dim (int): Output dimensionality (e.g., number of classes).
        dropout_rate (float): Dropout probability for each MambaBlock.
        use_glu (bool): If True, each block will include a Linear->GLU stage
                        that preserves model_dim.
    """

    def __init__(
        self,
        num_blocks: int,
        model_dim: int,
        data_dim: int,
        label_dim: int,
        dropout_rate: float = 0.1,
        use_glu: bool = False,
    ):
        super().__init__()

        # Use PyTorch's built-in Embedding
        self.embedding = nn.Embedding(num_embeddings=data_dim, embedding_dim=model_dim)

        # Create multiple MambaBlocks
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    model_dim=model_dim, dropout_rate=dropout_rate, use_glu=use_glu
                )
                for _ in range(num_blocks)
            ]
        )

        # The final linear projection remains (model_dim -> label_dim)
        self.linear = nn.Linear(model_dim, label_dim)

    def mask_grads(self):
        """
        This method is included for consistency with other models.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of StackedMamba.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
                              containing integer token IDs (if used with nn.Embedding).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, label_dim).
                          If a single-vector output is desired (e.g. for classification),
                          additional pooling or indexing may be required
                          before the final linear layer or after its output.
        """
        # Embedding: (batch_size, seq_len, model_dim)
        x = self.embedding(x)

        # Pass through each MambaBlock
        for block in self.blocks:
            x = block(x)

        # Final projection: (batch_size, seq_len, label_dim)
        return self.linear(x)
