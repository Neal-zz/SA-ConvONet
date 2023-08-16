import torch
import torch.optim as optim
import os
import yaml
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
# load model
filename = os.path.join(cfg['test']['model_path'], cfg['test']['model_file'])
state_dict = torch.load(filename).get('model')
model.load_state_dict(state_dict)

# Generator: generation.Generator3D
generator = conv_onet.config.get_generator(model, cfg, device=device)

# out/
mesh_dir = os.path.join(generation_dir, 'meshes')
log_dir = os.path.join(generation_dir, 'log')
# Create directories if necessary
if not os.path.exists(mesh_dir):
    os.makedirs(mesh_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# logger
logger = SummaryWriter(log_dir)

# 一些参数
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
trainer = conv_onet.training.Trainer(model, optimizer, device=device)

# 保存模型
checkpoint_io = CheckpointIO(cfg['training']['out_dir'], model=model, optimizer=optimizer)

# range(0, 720)
for iter in range(0, n_iter):
    # 只有一个 model 所以只循环一次
    for it, data in enumerate(test_loader):
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

        # (data, 6, 1536, 512, 0.1)
        loss = trainer.sign_agnostic_optim_step(data, batch_size, npoints1, npoints2, sigma)
        
        logger.add_scalar('test_optim/loss', loss, iter)
        print('[It %02d] iter_ft=%03d, loss=%.4f' % (it, iter, loss))
        
        if (iter + 1) % n_step == 0:
            lr = lr * lr_decay
            for g in optimizer.param_groups:
                g['lr'] = lr
                trainer = conv_onet.training.Trainer(
                    model, optimizer, device=device
                )
            for th in thres_list:
                generate_mesh_func(iter, th=th, suffix=f"th{th}")
            print('Saving checkpoint')
            checkpoint_io.save('model_%03d.pt' % (iter), epoch_it=0, it=iter,
                                loss_val_best=loss)

for th in thres_list:
    generate_mesh_func(n_iter, is_final=True, th=th, suffix=f"th{th}")

print('optimization finish.')