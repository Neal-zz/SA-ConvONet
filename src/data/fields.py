import os
import numpy as np

class PointCloudField(object):
    '''随机采样 3w 个点'''

    def __init__(self, file_name, transform=None):
        self.file_name = file_name  # pointcloud.npz
        self.transform = transform  

    def load(self, model_path):
        file_path = os.path.join(model_path, self.file_name)
        pointcloud_dict = np.load(file_path)
        points = pointcloud_dict['points'].astype(np.float32)
        data = points
        if self.transform is not None:
            data = self.transform(data)
        return data

