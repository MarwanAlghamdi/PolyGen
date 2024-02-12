# Import
import math
import unittest
import torch
import torch.nn as nn
from torch.nn import (MultiheadAttention,
                      Linear,
                      Dropout,
                      LayerNorm,
                      TransformerEncoderLayer)
import pytorch_lightning as pl


########################################################################################################################

# PolyGen Encoder
class Polygen_EncoderLayer(TransformerEncoderLayer):
    """
    PolyGen Encoder Layer as in the paper
    """

    def __init__(self,
                 dim_model: int = 256,
                 dim_feedforward: int = 1024,
                 nHead: int = 4,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 return_zero: bool = True,
                 ):
        """
        :param dim_model: size of embedding vectors
        :param dim_feedforward: size of feedforward layer
        :param nHead: number of heads in multi-head attention
        :param dropout: dropout rate
        :param activation: activation function
        :param return_zero: (Alpha) if True, scale residuals with zero initializations.
        """

        # Override the init function of TransformerEncoderLayer
        super().__init__(d_model=dim_model,
                         nhead=nHead,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         activation=activation)

        self.self_attn = MultiheadAttention(embed_dim=dim_model, num_heads=nHead, dropout=dropout)

        self.linear1 = Linear(dim_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = LayerNorm(dim_model)

        self.activation = nn.ReLU()

        self.rezero = return_zero
        self.alpha = nn.Parameter(data=torch.Tensor([0.0]))
        self.beta = nn.Parameter(data=torch.Tensor([0.0]))