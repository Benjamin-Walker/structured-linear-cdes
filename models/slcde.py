import torch
import torch.nn as nn
import torch.nn.functional as F


def hadamard_matrix(order):
    if order & (order - 1) != 0:
        raise ValueError("Order must be a power of 2 for Sylvester's construction.")
    if order == 1:
        return torch.tensor([[1]], dtype=torch.float32)

    H = hadamard_matrix(order // 2)
    return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)


class LinearCDE(nn.Module):
    """
    A linear controlled differential equation (CDE) model with recurrent-like updates.

    At each time step:
      - A matrix A(X) (from vf_A) is applied to the previous hidden state y.
      - A vector B(X) (from vf_B) is added.
      - The time increment is included as an additional input channel.

    We apply a sparsity mask to vf_A's weights during initialization and zero out the
    masked gradients so those weights remain zero.

    Args:
        input_dim (int): Dimensionality of the input features at each time step.
        hidden_dim (int): Hidden dimension for the recurrent state.
        output_dim (int): Dimensionality of the final output for each time step.
        sparsity (float): Probability of keeping a weight (i.e., 1.0 = no sparsity,
                          0.0 = all weights zero).
        init_std (float): Standard deviation for normal initialization. Different
                          layers have different scaled std in this implementation.

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len, output_dim)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        sparsity=1.0,
        init_std=1.0,
        diagonal=False,
        fwht=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.diagonal = diagonal
        self.fwht = fwht
        if self.fwht:
            self.hadamard = hadamard_matrix(hidden_dim).to(torch.device("cuda")) / (
                hidden_dim**0.5
            )
        else:
            self.hadamard = None

        # Define linear layers
        self.init_layer = nn.Linear(input_dim, hidden_dim, bias=True)
        if self.diagonal:
            self.vf_A = nn.Linear(input_dim + 1, hidden_dim, bias=False)
        else:
            self.vf_A = nn.Linear(input_dim + 1, hidden_dim * hidden_dim, bias=False)
            nn.init.normal_(
                self.vf_A.weight, mean=0.0, std=init_std / (hidden_dim**0.5)
            )
        self.vf_B = nn.Linear(input_dim + 1, hidden_dim, bias=False)

        # Apply custom weight initialization
        nn.init.normal_(self.init_layer.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.init_layer.bias, mean=0.0, std=init_std)
        # Scaled by sqrt(hidden_dim) to reduce variance
        nn.init.normal_(self.vf_B.weight, mean=0.0, std=init_std)

        # Register a buffer to store the mask on the same device as the module
        self.register_buffer("mask", self._sparse_mask(sparsity))

        # Zero out certain weights according to mask (only once at init)
        if not self.diagonal:
            with torch.no_grad():
                self.vf_A.weight *= self.mask

    def _sparse_mask(self, sparsity: float) -> torch.Tensor:
        """
        Returns a boolean mask for vf_A, shaped to match vf_A.weight:
          [hidden_dim * hidden_dim, input_dim + 1].
        Values in the mask are True with probability 'sparsity'.
        """
        mask = (
            torch.rand(self.hidden_dim * self.hidden_dim, self.input_dim + 1) < sparsity
        )
        return mask

    def mask_grads(self):
        """
        Applies the same mask to the gradients of vf_A.weight to keep them zero.
        This preserves the initially zeroed-out weights.
        """
        if self.vf_A.weight.grad is not None and not self.diagonal:
            self.vf_A.weight.grad *= self.mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Adds a uniform time channel (1 / (seq_len - 1)).
          2. Recurrently updates the hidden state y using A(X)y + B(X) at each step.
          3. Stores hidden states and applies a final linear layer.

        Args:
            X (torch.Tensor): shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = X.shape
        if seq_len < 2:
            # If seq_len == 1, the time increment 1/(seq_len-1) would be invalid.
            # Raise an error or handle as needed.
            raise ValueError(
                "Sequence length must be at least 2 for uniform time increments."
            )

        # Add increments of time as first channel
        ts = torch.full((batch_size, seq_len, 1), 1.0 / (seq_len - 1), device=X.device)
        inp = torch.cat((ts, X), dim=-1)  # shape: (batch_size, seq_len, input_dim+1)

        # Initialize the hidden state
        x0 = X[:, 0, :]  # shape: (batch_size, input_dim)
        y0 = self.init_layer(x0)  # shape: (batch_size, hidden_dim)

        # Prepare a tensor to store all hidden states
        ys = torch.zeros(batch_size, seq_len, self.hidden_dim, device=X.device)
        ys[:, 0] = y0
        y = y0

        # Precompute B offsets for steps 1..(seq_len-1)
        Bs = self.vf_B(inp[:, 1:])  # shape: (batch_size, seq_len-1, hidden_dim)

        # Recurrently compute the hidden states
        for i in range(1, seq_len):
            # A for this time step: shape = (batch_size, hidden_dim*hidden_dim)
            A = self.vf_A(inp[:, i])
            if self.diagonal:
                state_transition = (3**0.5) * torch.tanh(A) * y
                if self.fwht:
                    state_transition = torch.einsum(
                        "ij,bj->bi", self.hadamard, state_transition
                    )
                state_transition = state_transition + Bs[:, i - 1]
            else:
                state_transition = (
                    torch.einsum(
                        "bij,bj->bi",
                        A.view(-1, self.hidden_dim, self.hidden_dim),
                        y,
                    )
                    + Bs[:, i - 1]
                )
            y = y + state_transition
            ys[:, i] = y

        return ys


class LinearCDEBlock(nn.Module):
    """
    A residual block wrapping LinearCDE. Includes:
      1. LayerNorm on the input
      2. LinearCDE forward pass
      3. Residual connection
      4. (Optional) a Linear→GLU stage
      5. Dropout

    The output dimension of LinearCDE is the same as the input dimension
    to preserve shape for the residual.

    Args:
        input_dim (int): Dimensionality of the input (and thus output) features.
        hidden_dim (int): Hidden dimension inside the LinearCDE.
        init_std (float): Standard deviation for weight initialization in LinearCDE.
        sparsity (float): Probability for retaining a weight in vf_A of LinearCDE.
        dropout_rate (float): Dropout probability applied after the residual addition.

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len, input_dim)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        init_std=1.0,
        sparsity=1.0,
        dropout_rate=0.1,
        use_glu: bool = False,
        diagonal=False,
        fwht=False,
    ):
        super().__init__()
        self.LCDE = LinearCDE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            init_std=init_std,
            sparsity=sparsity,
            diagonal=diagonal,
            fwht=fwht,
        )
        self.norm = nn.LayerNorm(input_dim)

        # Optional GLU stage
        self.use_glu = use_glu
        if self.use_glu:
            # Expand from input_dim -> 2*input_dim for GLU gating
            self.glu_linear = nn.Linear(input_dim, 2 * input_dim)
        else:
            self.glu_linear = None

        self.drop = nn.Dropout(p=dropout_rate)

    def mask_grads(self):
        """
        Passes gradient masking down to the LinearCDE module.
        """
        self.LCDE.mask_grads()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            1. Compute LCDE on input
            2. Optionally apply GLU
            3. Add residual skip connection
            4. LayerNorm
            5. Dropout

        Args:
            X (torch.Tensor): shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: shape (batch_size, seq_len, input_dim)
        """
        ys = self.LCDE(X)  # shape: (batch_size, seq_len, input_dim)
        ys = ys + X  # residual skip

        # Optional GLU (dimension remains input_dim)
        if self.use_glu:
            ys_glu = self.glu_linear(ys)  # shape: (batch_size, seq_len, 2*input_dim)
            ys_glu = F.glu(ys_glu, dim=-1)  # shape: (batch_size, seq_len, input_dim)
            ys = ys + ys_glu  # residual skip

        ys = self.norm(ys)
        ys = self.drop(ys)  # dropout

        return ys


class StackedLCDE(nn.Module):
    """
    Stacks multiple LinearCDEBlocks, preceded by an embedding (for token IDs or similar),
    and followed by a final linear layer.

    Args:
        num_blocks (int): Number of LinearCDEBlocks to stack.
        hidden_dim (int): Hidden dimension used in each LinearCDEBlock.
        data_dim (int): Size of the input token (or feature) space if using an embedding.
        embedding_dim (int): Dimensionality of the embeddings.
        label_dim (int): Size of the final output dimension (e.g., number of classes).
        init_std (float): Standard deviation for the initialization in each block.
        sparsity (float): Probability for retaining a weight in vf_A of each block.
        dropout_rate (float): Dropout probability applied in each block after the residual.

    Shape:
        - Input: (batch_size, seq_len) if the input is token IDs (for the Embedding).
        - Output: (batch_size, seq_len, label_dim)
    """

    def __init__(
        self,
        num_blocks: int,
        model_dim: int,
        data_dim: int,
        label_dim: int,
        init_std: float = 1.0,
        sparsity: float = 1.0,
        dropout_rate: float = 0.01,
        use_glu: bool = False,
        diagonal=False,
        fwht=False,
        second_embedding=False,
    ):
        super().__init__()
        self.second_embedding = second_embedding
        embedding_dim = model_dim // 2 if second_embedding else model_dim
        self.embedding = nn.Embedding(data_dim, embedding_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, embedding_dim)

        # Build the stack of LCDE blocks
        self.blocks = nn.ModuleList(
            [
                LinearCDEBlock(
                    input_dim=embedding_dim,
                    hidden_dim=model_dim,
                    init_std=init_std,
                    sparsity=sparsity,
                    dropout_rate=dropout_rate,
                    use_glu=use_glu,
                    diagonal=diagonal,
                    fwht=fwht,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final projection: from embedding_dim -> label_dim
        self.linear = nn.Linear(embedding_dim, label_dim)

    def mask_grads(self):
        """
        Masks the gradients of vf_A weights in each block to preserve zeros.
        """
        for block in self.blocks:
            block.mask_grads()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the stacked model:
            1. Embed input X
            2. Pass through each LCDE block
            3. Apply final linear projection

        Args:
            X (torch.Tensor): If using embedding, shape (batch_size, seq_len).
                              If already an embedding, shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: shape (batch_size, seq_len, label_dim)
        """
        # Step 1: Embedding
        X = self.embedding(X)  # -> (batch_size, seq_len, embedding_dim)

        # Step 2: Pass through each LCDE block
        for block in self.blocks:
            X = block(X)  # (batch_size, seq_len, embedding_dim)

        # Step 3: Project to label_dim
        return self.linear(X)  # (batch_size, seq_len, label_dim)
