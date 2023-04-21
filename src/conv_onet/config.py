from src.encoder import pointnet
from src.conv_onet import models, generation, decoder
from src import data
from torchvision import transforms

def get_model(cfg, device=None):

    dim = cfg['data']['dim']           # 3
    c_dim = cfg['model']['c_dim']      # 32
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']   # 0.1
    
    # (3,32,0.1,decoder_kwargs)
    decoderObj = decoder.LocalDecoder(
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    # (3,32,0.1,encoder_kwargs)
    encoderObj = pointnet.LocalPoolPointnet(
        dim=dim, c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )

    model = models.ConvolutionalOccupancyNetwork(
        decoderObj, encoderObj, device=device
    )

    return model


def get_generator(model, cfg, device):

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test_optim']['threshold'][1],             # 0.2
        resolution0=cfg['generation']['resolution_0'],           # 32
        upsampling_steps=cfg['generation']['upsampling_steps'],  # 2
        padding=cfg['data']['padding'],                          # 0.1
    )
    return generator

def get_dataset(mode, cfg):

    # fields 每次随机采样 3w 个点
    transform = transforms.Compose([
        data.SubsamplePointcloud(cfg['data']['pointcloud_n'])
    ])
    inputs_field = data.PointCloudField(
        cfg['data']['pointcloud_file'], transform
    )
    fields = {}
    fields['inputs'] = inputs_field

    # data/demo
    dataset = data.Shapes3dDataset(cfg['data']['path'], fields)

    return dataset
