import torch
import torch.optim as optim
import os
import yaml
from tqdm import tqdm
from src import conv_onet
from tensorboardX import SummaryWriter

from src.checkpoints import CheckpointIO

# load config
with open('configs/demo_syn_room.yaml', 'r') as f:
    cfg = yaml.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cuda'
generation_dir = cfg['training']['out_dir']  # out

# Dataset
dataset = conv_onet.config.get_dataset('test', cfg)

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Model: models.ConvolutionalOccupancyNetwork
model = conv_onet.config.get_model(cfg, device=device)

# Generator: generation.Generator3D
generator = conv_onet.config.get_generator(model, cfg, device=device)

# 只有一个 model 所以只循环一次
for it, data in enumerate(tqdm(test_loader)):
    # out/
    mesh_dir = os.path.join(generation_dir, 'meshes')
    log_dir = os.path.join(generation_dir, 'log')

    # Create directories if necessary
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = SummaryWriter(log_dir)

    # load model
    filename = os.path.join(cfg['test']['model_path'], cfg['test']['model_file'])
    state_dict = torch.load(filename).get('model')
    model.load_state_dict(state_dict)

    def generate_mesh_func(iter, is_final=False, th=0.2, suffix='th0.2'):
        # Generate
        generator.threshold = th
        model.eval()
        mesh = generator.generate_mesh(data)

        # Write output
        if not is_final:
            mesh_out_file = os.path.join(mesh_dir, 'iter%d_%s.off' % (iter, suffix))
        else:
            mesh_out_file = os.path.join(mesh_dir, 'final_%s.off' % (suffix))
        mesh.export(mesh_out_file)
    
    # Intialize training using pretrained model, and then optimize network parameters for each observed input.
    lr = cfg['test_optim']['learning_rate']          # 0.00003
    lr_decay = cfg['test_optim']['decay_rate']       # 0.3
    n_iter = cfg['test_optim']['n_iter']             # 720
    n_step = cfg['test_optim']['n_step']             # 300
    batch_size = cfg['test_optim']['batch_size']     # 6
    npoints1 = cfg['test_optim']['npoints_surf']     # 1536
    npoints2 = cfg['test_optim']['npoints_nonsurf']  # 512
    sigma = cfg['test_optim']['sigma']               # 0.1
    thres_list = cfg['test_optim']['threshold']      # [0.15, 0.2, 0.25] 三个阈值用来判读是否为 surface
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = conv_onet.training.Trainer(
        model, optimizer, device=device
    )
    
    # Generate results before test-time optimization (results of pretrained ConvONet)
    for th in thres_list:
        generate_mesh_func(0, th=th, suffix=f"th{th}")

print('finish.')