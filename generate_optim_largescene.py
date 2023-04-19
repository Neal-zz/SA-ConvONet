import torch
import torch.optim as optim
import os
import shutil
import argparse
from tqdm import tqdm
import time, datetime
from collections import defaultdict
import pandas as pd
from src import config
from src import conv_onet
from tensorboardX import SummaryWriter

# load config
cfg = config.load_config('configs/demo_syn_room.yaml', 'configs/default.yaml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cuda'
generation_dir = cfg['training']['out_dir']  # out
input_type = cfg['data']['input_type']       # pointcloud
vis_n_outputs = cfg['generation']['vis_n_outputs']  # 2

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = conv_onet.config.get_model(cfg, device=device, dataset=dataset)

# Generator
generator = conv_onet.config.get_generator(model, cfg, device=device)

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# 只有一个 model 所以只循环一次
for it, data in enumerate(tqdm(test_loader)):
    # out/
    mesh_dir = os.path.join(generation_dir, 'meshes')
    log_dir = os.path.join(generation_dir, 'log')

    # model information
    idx = data['idx'].item() # 0
    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    modelname = model_dict['model']                  # room9_noroof
    category_id = model_dict.get('category', 'n/a')  # yinshe_dataset_norm

    # Create directories if necessary
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = SummaryWriter(log_dir)

    # load pretrained model
    filename = os.path.join(cfg['test']['model_path'], cfg['test']['model_file'])
    state_dict = torch.load(filename).get('model')

    def generate_mesh_func(modename, iter, is_final=False, th=0.4, suffix='th0.4'):
        # Generate
        generator.threshold = th
        model.eval()
        out = generator.generate_mesh(data)

        # Get statistics
        mesh, _ = out

        # Write output
        if not is_final:
            if iter > 0:
                mesh_out_file = os.path.join(mesh_dir, '%s_iter%d_%s.off' % (modelname, iter, suffix))
            else:
                mesh_out_file = os.path.join(mesh_dir, '%s.off' % (modelname))
        else:
            mesh_out_file = os.path.join(mesh_dir, '%s_final_%s.off' % (modelname, suffix))
        mesh.export(mesh_out_file)

    # Generate results before test-time optimization (results of pretrained ConvONet)
    th = cfg['test']['threshold']  # 0.2
    generate_mesh_func(modelname, 0, th=th, suffix=f"th{th}")
    
    # Intialize training using pretrained model, and then optimize network parameters for each observed input.
    lr = cfg['test_optim']['learning_rate']          # 0.00003
    lr_decay = cfg['test_optim']['decay_rate']       # 0.3
    n_iter = cfg['test_optim']['n_iter']             # 720
    n_step = cfg['test_optim']['n_step']             # 300
    batch_size = cfg['test_optim']['batch_size']     # 6
    thres_list = cfg['test_optim']['threshold']      # [0.4, 0.45, 0.5] 三个阈值用来判读是否为 surface
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    scene_index = int(data["idx"])
    # range(0, 720)
    for iter in range(0, n_iter):

        # load cropped patches into a batch
        dataset.split = "train"
        crop_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices=[scene_index]), batch_size=1)
        crop_data_lst = list()
        for _ in range(batch_size):
            added = False
            while not added:
                try:
                    crop_data_lst += [data for data in crop_loader] 
                    added = True
                except Exception as e:
                    print(f"encountering error: {e}")
                    pass

        cat_fn = lambda the_key: torch.cat([d[the_key] for d in crop_data_lst], dim=0) if isinstance(crop_data_lst[0][the_key], torch.Tensor) else {_key: torch.cat([d[the_key][_key] for d in crop_data_lst], dim=0) for _key in crop_data_lst[0][the_key].keys()}
        crop_data = dict(pointcloud_crop=True, **{k: cat_fn(k) for k in 
                                                  ["points", "points.occ", "points.normalized", "inputs", "inputs.ind", "inputs.mask"]})
        dataset.split = "test"

        loss = trainer.sign_agnostic_optim_cropscene_step(crop_data, state_dict)
        logger.add_scalar('test_optim/loss', loss, iter)
        
        print('[It %02d] iter_ft=%03d, loss=%.4f' % (it, iter, loss))
        if (iter + 1) % n_step == 0:
            lr = lr * lr_decay
            for g in optimizer.param_groups:
                g['lr'] = lr
            trainer = config.get_trainer(model, optimizer, cfg, device=device)
            for th in thres_list:
                generate_mesh_func(modelname, iter, th=th, suffix=f"th{th}")

    for th in thres_list:
        generate_mesh_func(modelname, n_iter, is_final=True, th=th, suffix=f"th{th}")


print('end.')
