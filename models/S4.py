"""
Implementation of the S4 model taken from https://github.com/state-spaces/s4
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p)
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(
            dt_min
        )

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(
        self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args
    ):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class S4Block(nn.Module):
    """
    A single S4 block that applies:
      1. S4D module
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
        self.s4d = S4D(d_model=model_dim, transposed=False)
        self.norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(p=dropout_rate)

        self.use_glu = use_glu
        if self.use_glu:
            self.post_linear = nn.Linear(model_dim, 2 * model_dim)
        else:
            self.post_linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the S4Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape (batch_size, seq_len, model_dim).
        """

        # S4 module
        y = self.s4d(x)
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


class StackedS4(nn.Module):
    """
    A stack of multiple S4Blocks, preceded by an embedding layer
    and followed by a linear projection.

    Args:
        num_blocks (int): Number of S4Blocks to stack.
        model_dim (int): Dimensionality of embeddings and S4 blocks.
        data_dim (int): Size of the vocabulary (if input is token IDs).
        label_dim (int): Output dimensionality (e.g., number of classes).
        dropout_rate (float): Dropout probability for each S4Block.
        use_glu (bool): If True, each block will include a Linear->GLU stage
                        that preserves model_dim.
        second_embedding (bool): If True, the model will expect two input
                                    token IDs and use two separate embeddings.
    """

    def __init__(
        self,
        num_blocks: int,
        model_dim: int,
        data_dim: int,
        label_dim: int,
        dropout_rate: float = 0.1,
        use_glu: bool = False,
        second_embedding: bool = False,
    ):
        super().__init__()

        self.second_embedding = second_embedding
        embedding_dim = model_dim // 2 if second_embedding else model_dim
        self.embedding = nn.Embedding(data_dim, embedding_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, embedding_dim)

        # Create multiple S4Blocks
        self.blocks = nn.ModuleList(
            [
                S4Block(model_dim=model_dim, dropout_rate=dropout_rate, use_glu=use_glu)
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
        Forward pass of StackedS4.

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
        if not self.second_embedding:
            x = self.embedding(x)
        else:
            x = torch.cat(
                [self.embedding(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )

        # Pass through each S4Block
        for block in self.blocks:
            x = block(x)

        # Final projection: (batch_size, seq_len, label_dim)
        return self.linear(x)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding for the current step
        if self.second_embedding:
            x = torch.cat(
                [self.embedding(x[:, 0].long()), self.embedding2(x[:, 1].long())],
                dim=-1,
            )
        else:
            x = self.embedding(x)

        for block in self.blocks:
            x = block.step(x)

        # Final projection for the step
        return self.linear(x)
