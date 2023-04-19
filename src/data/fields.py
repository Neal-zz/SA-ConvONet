import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field
from src.common import coord2index, normalize_coord


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class PointsField(Field):
    '''
    加载点云
    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name      # room9_noroof
        self.transform = transform      # 随机采样 2048 个点
        self.unpackbits = unpackbits    # True
        self.multi_files = multi_files  # null

    def load(self, model_path, idx, category):

        if self.multi_files is None:
            # 路径是正确的
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.transform is not None:
            data = self.transform(data)
        return data

class PointCloudField(Field):
    ''' 
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name      # pointcloud.npz
        self.transform = transform      # 

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''

        file_path = os.path.join(model_path, self.file_name)
        
        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        
        data = {
            None: points,
            'normals': normals,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

