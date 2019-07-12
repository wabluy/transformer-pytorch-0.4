import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_positional_encoding(n_position, hidden_size, padding_idx=None):
    """Sinusoidal Positional_Encoding."""

    def cal_angle(pos, i):
        return pos / np.power(10000, 2 * (i // 2) / hidden_size)

    def get_pos_angle_vector(pos):
        return [cal_angle(pos, i) for i in range(hidden_size)]  # (num_units, )

    position_enc = np.array([get_pos_angle_vector(pos) for pos in range(n_position)])  # (n_position, num_units)

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        position_enc[padding_idx] = 0

    position_enc = torch.FloatTensor(position_enc).to(device)
    return position_enc


def label_smoothing(labels, epsilon=0.1):
    K = labels.size()[-1]
    return ((1 - epsilon) * labels) + (epsilon / K)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, causality=False):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.causality = causality
        self.Q_fc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.K_fc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.V_fc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.out_dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, Q, K, V, key_pad_masks=None, query_pad_masks=None):
        """Forward function.

        Arguments:
            q: Tensor with shape=(batch_size, T_q, C)
            k: Tensor with shape=(batch_size, T_k, C)
            v: Tensor with shape=(batch_size, T_k, C)
            key_pad_masks: Tensor with value 1 where k is PAD. Shape=(batch_size, T_q, T_k)
            query_pad_masks: Tensor with value 1 where q is PAD. Shape=(batch_size, T_q, T_k)
        Returns:
            outputs: Tensor with shape=(batch_size, T_q, C)
        """
        # Linear projections
        Q = self.Q_fc(Q)  # (N, T_q, C)
        K = self.K_fc(K)  # (N, T_k, C)
        V = self.V_fc(V)  # (N, T_k, C)

        # Split to multi-head
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Dot-Product
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scaled
        outputs = outputs / (K_.shape[2] ** 0.5)  # (h*N, T_q, T_k)

        # Key Masking
        key_pad_masks = key_pad_masks.repeat(self.num_heads, 1, 1)  # (h*N, T_q, T_k)
        outputs.masked_fill(key_pad_masks, -np.inf)

        # Casuality Masking for Target language self-attention
        if self.causality:
            subsequent_mask = torch.triu(
                torch.ones(outputs.shape[1], outputs.shape[2], dtype=torch.uint8), diagonal=1).to(device)  # (T_q, T_k)
            subsequent_mask = subsequent_mask.unsqueeze(0).repeat(outputs.shape[0], 1, 1)  # (h*N, T_q, T_k)
            outputs.masked_fill(subsequent_mask, -np.inf)

        # Attention
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_pad_masks = query_pad_masks.repeat(self.num_heads, 1, 1)  # (h*N, T_q, T_k)
        outputs.masked_fill(query_pad_masks, 0)  # (h*N, T_q, T_k)

        # Dropout (But I'm not sure whether it is necessary.)
        outputs = self.out_dropout(outputs)  # (h*N, T_q, T_k)

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        outputs += Q  # (N, T_q, C)

        # Layer Normalization
        outputs = self.layer_norm(outputs)  # (N, T_q, C)

        return outputs


class FeedForward(nn.Module):
    def __init__(self, hidden_size, hiddens=[2048, 512]):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hiddens[0]), nn.ReLU())
        self.fc2 = nn.Linear(hiddens[0], hiddens[1])
        self.LN = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        """inputs: Tensor of shape=(batch_size, T, C)"""
        outputs = self.fc1(inputs)
        outputs = self.fc2(outputs)  # (N, T, C)

        # Residual connection
        outputs += inputs

        # Layer Normalization
        outputs = self.LN(outputs)

        return outputs
