from src.encoder import pointnet
from src.conv_onet import models, generation, decoder
from src import data
from torchvision import transforms

def get_model(cfg, device=None, dataset=None, **kwargs):

    dim = cfg['data']['dim']           # 3
    c_dim = cfg['model']['c_dim']      # 32
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']   # 0.1
    
    # simple_local(3,32,0.1,decoder_kwargs)
    decoderObj = decoder.LocalDecoder(
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    # pointnet_local_pool(3,32,0.1,encoder_kwargs)
    encoderObj = pointnet.LocalPoolPointnet(
        dim=dim, c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )

    model = models.ConvolutionalOccupancyNetwork(
        decoderObj, encoderObj, device=device
    )

    return model


def get_generator(model, cfg, device, **kwargs):

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


def get_dataset(mode, cfg):

    # data/demo
    dataset_folder = cfg['data']['path']

    # add inputs
    transform = transforms.Compose([
        data.SubsamplePointcloud(cfg['data']['pointcloud_n']),  # 随机采样 3w 个点
        data.PointcloudNoise(cfg['data']['pointcloud_noise'])   # 增加 0 噪声
    ])
    inputs_field = data.PointCloudField(
        cfg['data']['pointcloud_file'], transform  # 拿到 inputs 和 inputs.normals
    )
    fields = {}
    if inputs_field is not None:
        fields['inputs'] = inputs_field

    # dataset
    dataset = data.Shapes3dDataset(dataset_folder, fields)

    return dataset
