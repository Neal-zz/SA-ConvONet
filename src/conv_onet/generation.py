import torch
import numpy as np
import trimesh
from src.utils import libmcubes
from src.utils.libmise import MISE

# 由神经网路的输出，生成 mesh
class Generator3D(object):

    def __init__(self, model, threshold=0.2, device=None,
                resolution0=32, upsampling_steps=2, padding=0.1
                ):
        self.model = model.to(device)
        self.points_batch_size = 100000           # 10w
        self.threshold = threshold                # 0.2
        self.device = device                      # 'cuda'
        self.resolution0 = resolution0            # 32
        self.upsampling_steps = upsampling_steps  # 2
        self.padding = padding                    # 0.1

    # 函数接口
    def generate_mesh(self, data):
        self.model.eval()

        box_size = 1 + self.padding  # 1.1
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)  # logits（-1.386）
        inputs = data.get('inputs', torch.empty(1, 0)).to(self.device)    # 3w 个点
        
        with torch.no_grad():
            # 1,32,64,64,64
            c = self.model.encode_inputs(inputs)

        # MISE(32,2,-1.386)
        mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)
        # （33*33*33，3）从 0 到 128 间距为 4 等距离分布的点云
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # / 128.
            pointsf = points / mesh_extractor.resolution
            # +- 0.55
            pointsf = box_size * (pointsf - 0.5)
            pointsf = torch.FloatTensor(pointsf).to(self.device)
            # 调用 decoder，评估 pointsf 是否为内点
            values = self.eval_points(pointsf, c).cpu().numpy()
            values = values.astype(np.float64)
            # 如果 values 小于 threshold（内点），则网格进一步细分，否则就不必细分。
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()
        # 返回 129*129*129 的 values 矩阵
        value_grid = mesh_extractor.to_dense()

        mesh = self.extract_mesh(value_grid, c)
        return mesh

    def eval_points(self, p, c):
        # 每 10w 个为一组（只有一组）
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        # 每次拿 10w 个点
        for pi in p_split:
            # 1,35937,3
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, c).logits
                occ_hat = occ_hat.abs()  # 转换到 (0,inf)
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        occ_hat = torch.cat(occ_hats, dim=0)
        # +-inf
        return occ_hat

    def extract_mesh(self, occ_hat, c):
        n_x, n_y, n_z = occ_hat.shape  # 129, 129, 129
        occ_hat = occ_hat * -1.0  # 转换到 (-inf,0)
        box_size = 1 + self.padding    # 1.1
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)  # logits（-1.386）
        # threshold = np.log(1.-self.threshold) - np.log(self.threshold)
        threshold = threshold*-1.0
        
        
        # occ_hat 向外拓展一圈，并设为-1e6
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        # 提取 mesh，threshold 以上
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=None,
                               process=False)

        return mesh
