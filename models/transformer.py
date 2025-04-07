import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    """
    Pre-computes the sine/cosine (cos_theta, sin_theta) values needed
    for Rotary Position Embeddings (RoPE).

    For each head dimension 'head_dim', we store an 'inv_freq' vector:
         inv_freq[i] = 1 / (base^(2i / head_dim))
    Then for each position, we compute rotation angles = positions * inv_freq.

    The dimension is typically split in half for the actual rotation
    (the even-odd pairs in each head).
    """

    def __init__(self, num_heads, d_model, base=10000.0, max_len=500):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.base = base
        self.max_len = max_len

        # We usually assume head_dim is even so that we can split in half cleanly
        assert (
            self.head_dim % 2 == 0
        ), f"RoPE typically requires an even head_dim, but got {self.head_dim}."

        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        )
        # Register as buffer for easy device transfer
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute a maximum cos/sin buffer for [max_len].
        # This can be re-used if seq_len <= max_len.
        positions = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(
            1
        )  # shape: [max_len, 1]
        # shape: [max_len, half_dim]
        theta = positions * self.inv_freq.unsqueeze(0)
        # shape: [1, max_len, half_dim]
        cos_cached = torch.cos(theta).unsqueeze(0)
        sin_cached = torch.sin(theta).unsqueeze(0)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len, device=None):
        """
        Returns cos_theta, sin_theta for positions [0..seq_len-1].
        Shapes: (1, seq_len, half_dim), broadcastable to (B, nHeads, seq_len, half_dim).
        """
        if device is None:
            device = self.cos_cached.device

        if seq_len <= self.max_len:
            # Just slice the precomputed buffers
            cos_theta = self.cos_cached[:, :seq_len, :]
            sin_theta = self.sin_cached[:, :seq_len, :]
        else:
            # If seq_len > max_len, compute on the fly (less common)
            positions = torch.arange(
                seq_len, device=device, dtype=torch.float32
            ).unsqueeze(1)
            theta = positions * self.inv_freq.unsqueeze(0).to(device)
            cos_theta = torch.cos(theta).unsqueeze(0)
            sin_theta = torch.sin(theta).unsqueeze(0)

        return cos_theta.to(device), sin_theta.to(device)


def apply_rope(q, k, cos, sin):
    """
    q, k: shape (B, nHeads, T, head_dim)
    cos, sin: shape (1, seq_len, half_dim)

    We'll do the pairwise rotation in the first half of 'head_dim'
    (split head_dim => half_dim + half_dim).
    """
    bsz, n_heads, seq_len, head_dim = q.shape
    half_dim = head_dim // 2

    # Split them into [..., :half_dim] and [..., half_dim:]
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    # For broadcast, cos/sin must shape => [B, nHeads, seq_len, half_dim]
    cos = cos[:, :seq_len, :]  # (1, seq_len, half_dim)
    sin = sin[:, :seq_len, :]
    cos = cos.unsqueeze(1).expand(bsz, n_heads, seq_len, half_dim)
    sin = sin.unsqueeze(1).expand(bsz, n_heads, seq_len, half_dim)

    # rotate:
    #   q1_rot = q1*cos - q2*sin
    #   q2_rot = q2*cos + q1*sin
    q1_rot = q1 * cos - q2 * sin
    q2_rot = q2 * cos + q1 * sin
    k1_rot = k1 * cos - k2 * sin
    k2_rot = k2 * cos + k1 * sin

    q_rot = torch.cat([q1_rot, q2_rot], dim=-1)
    k_rot = torch.cat([k1_rot, k2_rot], dim=-1)
    return q_rot, k_rot


def custom_scaled_dot_product_attention(
    Q, K, V, attn_mask=None, dropout_p=0.0, training=False
):
    """
    Q, K, V: [B, nHeads, T, head_dim]
    attn_mask (optional): [B, nHeads, T, T] or broadcastable
    Returns:
        attn_output: [B, nHeads, T, head_dim]
        attn_weights: [B, nHeads, T, T]
    """
    B, nHeads, T, head_dim = Q.shape

    # 1) Scale dot products
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    # 2) Add mask/bias
    if attn_mask is not None:
        scores = scores + attn_mask  # shape must be broadcastable to [B, nHeads, T, T]

    # 3) Convert to probabilities
    attn_weights = F.softmax(scores, dim=-1)

    # 4) Apply dropout on attention weights if needed
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    # 5) Multiply by V
    attn_output = torch.matmul(attn_weights, V)  # [B, nHeads, T, head_dim]

    return attn_output, attn_weights


class CustomMultiheadAttention(nn.Module):
    """
    Multihead attention that supports a 4D (per-head) bias/mask,
    returning both attn_output and attn_weights.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout_p = dropout

    def forward(self, x, attn_mask=None, need_weights=True, rope_module=None):
        """
        x: [B, T, d_model]
        attn_mask (optional): [B, nHeads, T, T]
        rope_module: if not None, an instance of RotaryPositionalEncoding

        Returns:
            out: [B, T, d_model]
            attn_weights (opt): [B, nHeads, T, T]
        """
        B, T, _ = x.shape

        # 1) Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) Reshape to [B, nHeads, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_module is not None:
            cos, sin = rope_module(seq_len=T, device=x.device)
            Q, K = apply_rope(Q, K, cos, sin)

        # 3) Scaled-dot-product attention w/ possible 4D mask
        attn, attn_weights = custom_scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        # attn: [B, nHeads, T, head_dim]
        # attn_weights: [B, nHeads, T, T]

        # 4) Merge heads => [B, T, d_model]
        attn = attn.transpose(1, 2).reshape(B, T, self.d_model)

        # 5) Final linear
        out = self.W_o(attn)

        if not need_weights:
            return out
        else:
            return out, attn_weights


class MultiheadAttentionWithPositionEnc(nn.Module):
    def __init__(
        self, d_model, nhead, dropout, use_rope=True, max_len=500, need_weights=True
    ):
        """
        - use_rope: whether to use rotary position embeddings
        """
        super().__init__()
        self.use_rope = use_rope
        self.num_heads = nhead
        self.need_weights = need_weights

        self.attention = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

        if self.use_rope:
            self.rope = RotaryPositionalEncoding(nhead, d_model, max_len=max_len)

    def forward(self, x, input_pad_mask=None, need_weights=True):
        """
        x: [batch_size, seq_len, d_model]
        input_pad_mask: [B, T], True => real token, False => pad
        returns: (attn_output, attn_weights)  if need_weights=True
                 attn_output                  otherwise
        """
        B, T, _ = x.shape

        attn_mask = None

        # Combine with pad mask (mask out padded 'key' positions)
        if input_pad_mask is not None:
            # input_pad_mask is [B, T], True => real token
            # We want to produce a mask for the "key" dimension:
            # shape => [B, 1, 1, T], True => mask out
            pad_mask = ~input_pad_mask.unsqueeze(1).unsqueeze(2)
            # If attn_mask is None (no relative), create zeros first
            if attn_mask is None:
                attn_mask = torch.zeros(
                    (B, self.num_heads, T, T), device=x.device, dtype=x.dtype
                )
            # Now mask out the padded keys
            attn_mask = attn_mask.masked_fill(pad_mask, float("-inf"))

        # ---------------------------------------------------------------------
        # NEW: Build the causal mask so that tokens cannot attend to "future" positions
        # causal_mask[i, j] = True if j > i (we mask those).
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        # Expand to shape [1, 1, T, T] so it can broadcast
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]

        # If no existing mask, create a float-zero mask. Then mask out future positions.
        if attn_mask is None:
            attn_mask = torch.zeros(
                (B, self.num_heads, T, T), device=x.device, dtype=x.dtype
            )
            attn_mask = attn_mask.masked_fill(causal_mask, float("-inf"))
        else:
            attn_mask = attn_mask.masked_fill(causal_mask, float("-inf"))
        # ---------------------------------------------------------------------

        # Our CustomMultiheadAttention can accept an attn_mask of shape [B, nHeads, T, T]
        results = self.attention(
            x,
            attn_mask=attn_mask,
            need_weights=self.need_weights,
            rope_module=self.rope if self.use_rope else None,
        )

        return results


class PreNormTransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer with Pre-Layer Normalization (causal mask)."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        max_len,
        need_weights=True,
        use_rope=False,
    ):
        super(PreNormTransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiheadAttentionWithPositionEnc(
            d_model,
            nhead,
            dropout,
            use_rope=use_rope,
            max_len=max_len,
            need_weights=need_weights,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.need_weights = need_weights
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_pad_mask=None):
        """
        x: [B, T, d_model]
        input_pad_mask: [B, T] or None
        """
        if self.need_weights:
            attn_out, attn_weights = self.attention(self.norm1(x), input_pad_mask)
        else:
            attn_out = self.attention(self.norm1(x), input_pad_mask)

        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        data_dim,
        model_dim,
        label_dim,
        dropout_rate,
        num_heads=16,
        ff_hidden_dim=512,
        use_rope=True,
        max_len=1000,
        second_embedding=False,
    ):
        super(CausalTransformer, self).__init__()

        self.second_embedding = second_embedding
        embedding_dim = model_dim // 2 if second_embedding else model_dim

        self.embedding = nn.Embedding(data_dim, embedding_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, embedding_dim)

        self.use_rope = use_rope

        # Initialize absolute positional encoding if not using RoPE
        if not self.use_rope:
            self.positional_embedding = nn.Parameter(torch.zeros(max_len, model_dim))
            nn.init.xavier_uniform_(self.positional_embedding)

        self.decoder_layers = nn.ModuleList(
            [
                PreNormTransformerDecoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_hidden_dim,
                    dropout=dropout_rate,
                    activation="gelu",
                    use_rope=self.use_rope,
                    max_len=max_len,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, label_dim)

    def mask_grads(self):
        pass

    def forward(self, x, input_pad_mask=None):
        """
        x: shape [B, T] if single embedding, or [B, T, 2] if second_embedding=True
        input_pad_mask: [B, T], True => real token, False => pad
        """

        if not self.second_embedding:
            x = self.embedding(x)  # [B, T, model_dim]
        else:
            # If using second embedding then combine along last dim
            x = torch.cat(
                [self.embedding(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )  # [B, S, model_dim]

        # If using absolute positional encoding, add position embeddings
        if not self.use_rope:
            pos_embedding = self.positional_embedding[: x.size(1), :]
            x = x + pos_embedding.unsqueeze(0)

        # Pass through the decoder layers
        for layer in self.decoder_layers:
            x = layer(x, input_pad_mask)

        x = self.final_layer_norm(x)
        return self.linear(x)  # shape [B, T, label_dim]
