import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange, tqdm
import trimesh
from src.utils import libmcubes
from src.common import make_3d_grid, normalize_coord, add_key, coord2index
from src.utils.libsimplify import simplify_mesh
from src.utils.libmise import MISE
import time
import math

import open3d as o3d

counter = 0

# 由神经网路的输出，生成 mesh
class Generator3D(object):
    '''
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size  # 10w
        self.refinement_step = refinement_step      # 0
        self.threshold = threshold                  # 0.2
        self.device = device                        # 'cuda'
        self.resolution0 = resolution0              # 32
        self.upsampling_steps = upsampling_steps    # 2
        self.with_normals = with_normals            # false
        self.input_type = input_type                # pointcloud
        self.padding = padding                      # 0.1
        self.sample = sample                        # false
        self.simplify_nfaces = simplify_nfaces      # null
        self.vol_bound = vol_bound                  # None
        # None
        if vol_info is not None:
            self.input_vol, _, _ = vol_info
        
    def generate_mesh(self, data, return_stats=True):

        self.model.eval()
        device = self.device
        stats_dict = {}

        # 30000 个点云
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}
        t0 = time.time()
        
        # None
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            c = self.encode_crop(inputs, device)
        else:
            with torch.no_grad():
                c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0
        
        mesh = self.generate_from_latent(c, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh
    
    def generate_from_latent(self, c=None, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube
        Args:
            c (tensor): latent conditioned code c
        '''
        # -1.386
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        # 1.1
        box_size = 1 + self.padding
        
        # 2
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            # (32,2,-1.386)
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            # （33**3，3）从 0 到 128 间距为 4 等距离分布的点云
            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # / 128
                pointsf = points / mesh_extractor.resolution
                # +- 0.55
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # 评估 pointsf 是否为内点
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()  # 没有拿到任何点？？？？？

            value_grid = mesh_extractor.to_dense()

        mesh = self.extract_mesh(value_grid, c)
        return mesh

    def eval_points(self, p, c=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        # 10w
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        # 每次拿 10w 个点
        for pi in p_split:
            # false
            if self.input_type == 'pointcloud_crop':
                if self.vol_bound is not None: # sliding-window manner
                    occ_hat = self.predict_crop_occ(pi, c, vol_bound=vol_bound, **kwargs)
                    occ_hats.append(occ_hat)
                else: # entire scene
                    pi_in = pi.unsqueeze(0).to(self.device)
                    pi_in = {'p': pi_in}
                    p_n = {}
                    for key in c.keys():
                        # normalized to the range of [0, 1]
                        p_n[key] = normalize_coord(pi.clone(), self.input_vol, plane=key).unsqueeze(0).to(self.device)
                    pi_in['p_n'] = p_n
                    with torch.no_grad():
                        occ_hat = self.model.decode(pi_in, c, **kwargs).logits
                    occ_hats.append(occ_hat.squeeze(0).detach().cpu())
            else:
                # 1,35937,3
                pi = pi.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    occ_hat = self.model.decode(pi, c, **kwargs).logits
                occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def extract_mesh(self, occ_hat, c=None):
        ''' Extracts the mesh from the predicted occupancy grid.
        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        
        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh