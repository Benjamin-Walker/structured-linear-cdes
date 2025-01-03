import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self, num_blocks, data_dim, model_dim, label_dim, second_embedding=False
    ):
        super(LSTM, self).__init__()
        self.second_embedding = second_embedding
        embedding_dim = model_dim // 2 if second_embedding else model_dim
        self.embedding = nn.Embedding(data_dim, embedding_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, embedding_dim)
        self.lstm = nn.LSTM(
            model_dim,
            model_dim,
            num_layers=num_blocks,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.linear = nn.Linear(model_dim, label_dim)

    def mask_grads(self):
        pass

    def forward(self, x):
        if not self.second_embedding:
            x = self.embedding(x)
        else:
            x = torch.cat(
                [self.embedding1(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )
        x, _ = self.lstm(x)
        return self.linear(x)
