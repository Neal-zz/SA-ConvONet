import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet import decoder

class ConvolutionalOccupancyNetwork(nn.Module):

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        
        self.decoder = decoder.to(device)  # simple_local(3,32,0.1,decoder_kwargs)
        self.encoder = encoder.to(device)  # pointnet_local_pool(3,32,0.1,encoder_kwargs))

    def forward(self, p, inputs, sample=True, **kwargs):
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        c = self.encoder(inputs)
        return c

    def decode(self, p, c, **kwargs):
        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)  # 这是一种采样方法，好像没用？
        return p_r

