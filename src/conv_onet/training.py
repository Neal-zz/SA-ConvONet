import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
     add_key, #compute_iou, make_3d_grid,
)
from src.training import BaseTrainer
import numpy as np

# 用于 optimize
class Trainer(BaseTrainer):

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type    # pointcloud
        self.threshold = threshold      # 0.2
        self.eval_sample = eval_sample  # false

    # sign_agnostic_optim 接口
    def sign_agnostic_optim_step(self, data, state_dict, batch_size=16, npoints1=1024,
        npoints2=1024, sigma=0.1):

        self.model.train()
        self.optimizer.zero_grad()
        # (data, 6, 1536, 512, 0.1, None)
        loss = self.compute_sign_agnostic_loss(data, batch_size, npoints1, npoints2, sigma)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # 前向传播
    def compute_sign_agnostic_loss(self, data, batch_size, npoints1, npoints2, sigma):

        device = self.device
        inputs = data.get('inputs')  # (1,3w,3)
        batch_size, npoints1, npoints2 = int(batch_size), int(npoints1), int(npoints2)

        # (6,3w,3) 将 inputs 复制六次，用于 encode
        batch_inputs = inputs.expand(batch_size, inputs.size(1), inputs.size(2))
        c = self.model.encode_inputs(batch_inputs.to(device))
        
        # load query points and corresponding labels.
        batch_p = []
        batch_occ = []
        for i in range(batch_size):
            inputs_noise = sigma * np.random.normal(0, 1.0, size=inputs.cpu().numpy().shape)
            inputs_noise = torch.from_numpy(inputs_noise).type(torch.FloatTensor)
            inputs_noise = inputs + inputs_noise
            index1 = np.random.randint(inputs.size(1), size=npoints1)  # 生成 1536 个可重复的 index
            index2 = np.random.randint(inputs.size(1), size=npoints2)  # 生成 512 个可重复的 index
            # (1,2048,3)
            p = torch.cat([inputs[:, index1, :], inputs_noise[:, index2, :]], dim=1)
            # 0.5 代表是 surface，1 代表不是 surface。
            occ = torch.cat([torch.ones((1, npoints1), dtype=torch.float32)*0.5, torch.ones((1, npoints2), dtype=torch.float32)], dim=1)
            batch_p.append(p)
            batch_occ.append(occ)
        batch_p = torch.cat(batch_p, dim=0).to(device)
        batch_occ = torch.cat(batch_occ, dim=0).to(device)

        # General points
        kwargs = {}
        logits = self.model.decode(batch_p, c, **kwargs).logits
        logits = logits.abs()  # absolute value
        loss_i = F.binary_cross_entropy_with_logits(
            logits, batch_occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss

   