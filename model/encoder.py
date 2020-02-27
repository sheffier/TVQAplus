import torch.nn as nn
import torch.nn.functional as F

from typing import NamedTuple
from .position_encoding import PositionEncoding
from .cnn import DepthwiseSeparableConv
from .self_attention import MultiHeadedAttention


class StackedEncoderConf(NamedTuple):
    n_blocks: int = 7
    n_conv: int = 2
    kernel_size: int = 7
    hidden_size: int = 128
    dropout: float = 0.1
    num_heads: int = 4


class EncoderBlock(nn.Module):

    def __init__(self, n_conv, kernel_size=7, n_filters=128, dropout=0.1, num_heads=4):
        super(EncoderBlock, self).__init__()
        self.dropout = dropout
        self.n_conv = n_conv
        self.num_heads = num_heads

        self.position_encoding = PositionEncoding(n_filters=n_filters)

        self.layer_norm = nn.ModuleList([nn.LayerNorm(n_filters) for _ in range(n_conv)])
        self.final_layer_norm = nn.LayerNorm(n_filters)
        self.conv = nn.ModuleList([
            DepthwiseSeparableConv(in_ch=n_filters, out_ch=n_filters, k=kernel_size, relu=True)
            for _ in range(n_conv)])

        if self.num_heads != 0:
            self.multi_head_attn = MultiHeadedAttention(nh=num_heads, d_model=n_filters)
            self.attn_layer_norm = nn.LayerNorm(n_filters)

    def forward(self, x, mask):
        """
        :param x: (N, L, D)
        :param mask: (N, L)
        :return: (N, L, D)
        """
        outputs = self.position_encoding(x)  # (N, L, D)

        for i in range(self.n_conv):
            residual = outputs
            outputs = self.layer_norm[i](outputs)

            if i % 2 == 0:
                outputs = F.dropout(outputs, p=self.dropout, training=self.training)
            outputs = self.conv[i](outputs)
            outputs = outputs + residual

        if self.num_heads != 0:
            residual = outputs
            outputs = self.attn_layer_norm(outputs)
            outputs = self.multi_head_attn(outputs, mask=mask)
            outputs = outputs + residual

        return self.final_layer_norm(outputs)  # (N, L, D)


class StackedEncoder(nn.Module):
    def __init__(self, stack_enc_conf: StackedEncoderConf):
        super(StackedEncoder, self).__init__()

        self.n_blocks = stack_enc_conf.n_blocks
        self.stacked_encoderBlocks = nn.ModuleList([EncoderBlock(n_conv=stack_enc_conf.n_conv,
                                                                 kernel_size=stack_enc_conf.kernel_size,
                                                                 n_filters=stack_enc_conf.hidden_size,
                                                                 dropout=stack_enc_conf.dropout,
                                                                 num_heads=stack_enc_conf.num_heads)
                                                    for _ in range(stack_enc_conf.n_blocks)])

    def forward(self, x, mask):
        """
        :param x: # (N, L, D)
        :param mask:  # (N, L)
        :return:  (N, L, D)
        """
        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask)
        return x
