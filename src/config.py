import yaml
from torchvision import transforms
from src import data
from src import conv_onet


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def get_dataset(mode, cfg, return_idx=False):

    # data/demo
    dataset_folder = cfg['data']['path']
    # pointcloud
    input_type = cfg['data']['input_type']

    # get fields
    fields = conv_onet.config.get_data_fields(mode, cfg)  # test
    # add inputs
    if input_type == 'pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),  # 随机采样 3w 个点
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])   # 增加 0 噪声
        ])
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform  # 拿到 inputs 和 inputs.normals
        )
        if inputs_field is not None:
            fields['inputs'] = inputs_field
    else:
        raise ValueError('Invalid input type (%s)' % input_type)
    # true
    if return_idx:
        fields['idx'] = data.IndexField()  # 拿到 idx
    
    # dataset
    dataset = data.Shapes3dDataset(
        dataset_folder, fields,
        split=mode, cfg = cfg
    )

    return dataset

