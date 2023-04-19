import torch
import torch.distributions as dist
from torch import nn
import os
from src.encoder import encoder_dict
from src.conv_onet import models, training
from src.conv_onet import generation
from src import data
from src import config
from torchvision import transforms
import numpy as np

def get_model(cfg, device=None, dataset=None, **kwargs):
    '''
    '''
    decoder = cfg['model']['decoder']  # simple_local
    encoder = cfg['model']['encoder']  # pointnet_local_pool
    dim = cfg['data']['dim']           # 3
    c_dim = cfg['model']['c_dim']      # 32
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']   # 0.1
    
    # simple_local(3,32,0.1,decoder_kwargs)
    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    # pointnet_local_pool(3,32,0.1,encoder_kwargs)
    encoder = encoder_dict[encoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )

    model = models.ConvolutionalOccupancyNetwork(
        decoder, encoder, device=device
    )

    return model


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    vol_bound = None
    vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test_optim']['threshold'][1],             # 0.2
        resolution0=cfg['generation']['resolution_0'],           # 32
        upsampling_steps=cfg['generation']['upsampling_steps'],  # 2
        sample=cfg['generation']['use_sampling'],                # false
        refinement_step=cfg['generation']['refinement_step'],    # 0
        simplify_nfaces=cfg['generation']['simplify_nfaces'],    # null
        input_type = cfg['data']['input_type'],                  # pointcloud
        padding=cfg['data']['padding'],  # 0.1
        vol_info = vol_info,             # None
        vol_bound = vol_bound,           # NOne
    )
    return generator


def get_data_fields(mode, cfg):
    '''
    mode (str): the mode which is used
    cfg (dict): imported yaml config
    '''
    # 随机采样 2048 个点
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    # pointcloud
    input_type = cfg['data']['input_type']

    fields = {}
    # room9_noroof.ply
    if cfg['data']['points_file'] is not None:
        # True
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(  # 拿到 points 与 points.occ
                cfg['data']['points_file'],
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits']
            )
        else:
            fields['points'] = data.PatchPointsField(
                cfg['data']['points_file'],
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits']
            )

    # 'test'
    if mode in ('val', 'test'):
        # 有用吗？？？
        points_iou_file = cfg['data']['points_iou_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(  # 拿到 points_iou 与 points_iou.occ
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits']
                )
            else:
                fields['points_iou'] = data.PointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits']
                )

    return fields
