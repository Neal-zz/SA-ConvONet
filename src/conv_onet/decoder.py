import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_3d_coordinate

class LocalDecoder(nn.Module):

    def __init__(self, dim=3, c_dim=32, hidden_size=32, n_blocks=5, padding=0.1):
        super().__init__()
        self.c_dim = c_dim              # 32
        self.n_blocks = n_blocks        # 5
        self.padding = padding          # 0.1

        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
        ])
        self.fc_p = nn.Linear(dim, hidden_size)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, 1)
        self.actvn = F.relu

    def sample_grid_feature(self, p, c):
        # 将 p 从 (-0.55,0.55) 转换到 (0,1)
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # 转换到 (-1,1)
        # trilinear interpolation
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane):
        # 获取特征向量
        c = self.sample_grid_feature(p, c_plane['grid'])
        c = c.transpose(1, 2)

        p = p.float()  # xyz
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

