import torch
import torch.nn as nn
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


class xLSTM(nn.Module):
    def __init__(
        self,
        num_blocks,
        data_dim,
        model_dim,
        label_dim,
        dropout_rate,
        second_embedding=False,
        context_length=256,
        slstm_at=None,
    ):
        super(xLSTM, self).__init__()
        self.second_embedding = second_embedding
        embedding_dim = model_dim // 2 if second_embedding else model_dim
        self.embedding = nn.Embedding(data_dim, embedding_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, embedding_dim)

        # xLSTM configuration
        self.xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=model_dim,
            slstm_at=slstm_at or [1],
            dropout=dropout_rate,
        )

        self.xlstm = xLSTMBlockStack(self.xlstm_cfg)
        self.linear = nn.Linear(model_dim, label_dim)

    def mask_grads(self):
        pass

    def forward(self, x):
        if not self.second_embedding:
            x = self.embedding(x)
        else:
            x = torch.cat(
                [self.embedding(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )

        x = self.xlstm(x)

        return self.linear(x)
