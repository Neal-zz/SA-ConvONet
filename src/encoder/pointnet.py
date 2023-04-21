import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from torch_scatter import scatter_max, scatter_mean
from src.common import coordinate2index, normalize_3d_coordinate
from src.encoder.unet3d import UNet3D

class LocalPoolPointnet(nn.Module):

    def __init__(self, c_dim=32, dim=3, hidden_dim=32, unet3d_kwargs=None, 
                grid_resolution=64, padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim                # 32
        self.hidden_dim = hidden_dim      # 32
        self.reso_grid = grid_resolution  # 64
        self.padding = padding            # 0.1

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.unet3d = UNet3D(**unet3d_kwargs)

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid)
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)
        fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def pool_local(self, xy, index, c):
        fea_dim = c.size(2)  # 每个 point 的特征向量
        keys = xy.keys()  # 'grid'
        c_out = 0
        for key in keys:
            # 
            fea = scatter_max(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        # 将 p 从 (-0.55,0.55) 转换到 (0,1)
        coord = {}
        coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
        # 分辨率 64，每个 p 的 index
        index = {}
        index['grid'] = coordinate2index(coord['grid'], self.reso_grid)
        
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)

        fea = {}
        fea['grid'] = self.generate_grid_features(p, c)

        return fea

