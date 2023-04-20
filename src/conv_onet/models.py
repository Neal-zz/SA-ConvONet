import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet import decoder

class ConvolutionalOccupancyNetwork(nn.Module):

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        
        self.decoder = decoder.to(device)      # simple_local(3,32,0.1,decoder_kwargs)
        if encoder is not None:
            self.encoder = encoder.to(device)  # pointnet_local_pool(3,32,0.1,encoder_kwargs))
        else:
            self.encoder = None

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            print('encoder is none...')
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    
    # 这个函数确实有用
    def train(self, mode=True, freeze_norm=False, freeze_norm_affine=False):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)

        if freeze_norm:
            if self.encoder is not None:
                for m in self.encoder.modules():
                    if isinstance(m, nn.GroupNorm):
                        m.eval()
                        if freeze_norm_affine:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
            if self.decoder is not None:
                for m in self.decoder.modules():
                    if isinstance(m, nn.GroupNorm):
                        m.eval()
                        if freeze_norm_affine:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False