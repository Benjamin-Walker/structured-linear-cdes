import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class S4Recurrence(nn.Module):
    """
    Real S4D-style recurrence compatible with selective_scan_fn.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        dt_rank="auto",  # kept for API symmetry; unused here
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        device=None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        dt = torch.exp(
            torch.rand(self.d_model, device=self.device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.log_dt = nn.Parameter(torch.log(dt))

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        self.B = nn.Parameter(
            torch.empty(self.d_model, self.d_state, device=self.device)
        )
        self.C = nn.Parameter(
            torch.empty(self.d_model, self.d_state, device=self.device)
        )
        nn.init.xavier_normal_(self.B)
        nn.init.xavier_normal_(self.C)
        self.D = nn.Parameter(torch.ones(self.d_model, device=self.device))

    def forward(self, hidden_states):
        # x: (B, L, D) -> (B, D, L)
        x = rearrange(hidden_states, "b l d -> b d l").contiguous()

        A = -torch.exp(self.A_log.float())
        dt = self.log_dt.exp()[None, :, None].expand(
            x.shape[0], self.d_model, x.shape[2]
        )
        y = selective_scan_fn(
            x,
            dt,
            A,
            self.B.float(),
            self.C.float(),
            self.D.float(),
            z=None,
            delta_softplus=False,
            return_last_state=False,
        )
        return rearrange(y, "b d l -> b l d")


class MambaRecurrence(nn.Module):
    """
    Implements the Mamba recurrence layer for sequence modeling.

    Args:
        d_model (int): Dimension of the model (number of features).
        d_state (int): Dimension of the state space. Defaults to 16.
        dt_rank (int or str): Rank for the time-step parameterization. Defaults to
            "auto".
        dt_min (float): Minimum value for time-steps. Defaults to 0.001.
        dt_max (float): Maximum value for time-steps. Defaults to 0.1.
        dt_init (str): Initialization method for dt. Options are "constant" or "random".
            Defaults to "random".
        dt_scale (float): Scale factor for dt initialization. Defaults to 1.0.
        dt_init_floor (float): Floor value for initializing time-steps. Defaults to
            1e-4.
        device (torch.device, optional): Device to run the computations on. If None,
            uses CUDA if available.

    Forward Args:
        hidden_states (Tensor): Input tensor of shape (batch_size, sequence_length,
            d_model).

    Returns:
        Tensor: Output tensor of the same shape as input.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        device=None,
    ):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = nn.Linear(
            self.d_model,
            self.dt_rank + self.d_state * 2,
            bias=False,
            device=self.device,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_model, bias=True, device=self.device
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_model, device=self.device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this
        # one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_model, device=self.device)
        )  # Keep in fp32
        self.D._no_weight_decay = True

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        x = rearrange(hidden_states, "b l d -> b d l").contiguous()
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen).contiguous()
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        return rearrange(y, "b d l -> b l d")


class SSM(nn.Module):
    def __init__(
        self,
        recurrence_type: str,
        num_blocks: int,
        data_dim: int,
        model_dim: int,
        label_dim: int,
        dropout_rate: float = 0.1,
        second_embedding: bool = False,
        use_glu: bool = False,
        d_state: int = 16,
        dt_rank: str | int = "auto",
    ):
        """
        d_state: state dimension of the SSM (default 16).
        dt_rank: rank for dt parameterization ("auto" uses ceil(model_dim/16)).
        """
        super().__init__()
        self.second_embedding = second_embedding

        emb_dim = model_dim // 2 if second_embedding else model_dim
        self.embedding = nn.Embedding(data_dim, emb_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, emb_dim)

        if recurrence_type == "S6":
            recurrence_cls = MambaRecurrence
        elif recurrence_type == "S4":
            recurrence_cls = S4Recurrence
        else:
            raise ValueError(f"Unknown recurrence type: {recurrence_type}")

        self.blocks = nn.ModuleList(
            [
                recurrence_cls(model_dim, d_state=d_state, dt_rank=dt_rank)
                for _ in range(num_blocks)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_blocks)])

        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(model_dim, label_dim)

        self.use_glu = use_glu
        if use_glu:
            self.glu_projs = nn.ModuleList(
                [nn.Linear(model_dim, 2 * model_dim) for _ in range(num_blocks)]
            )
        else:
            self.glu_projs = nn.ModuleList(
                [nn.Linear(model_dim, model_dim) for _ in range(num_blocks)]
            )
        self.act = nn.GLU()

    def mask_grads(self):
        pass

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if not self.second_embedding:
            # x: (B, L)
            return self.embedding(x)  # -> (B, L, model_dim)
        else:
            # x: (B, L, 2)
            return torch.cat(
                [self.embedding(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )  # -> (B, L, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embed(x)  # (B, L, D=model_dim)
        for i, (blk, ln) in enumerate(zip(self.blocks, self.norms)):
            residual = x
            x = blk(x)
            h = self.glu_projs[i](x)
            h = self.act(h) if self.use_glu else torch.tanh(h)
            x = residual + x + h
            x = self.dropout(ln(x))
        return self.linear(x)  # (B, L, label_dim)
