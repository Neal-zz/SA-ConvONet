import numpy as np

class PointcloudNoise(object):
    '''随机增加 stddev level 的噪声'''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out

class SubsamplePointcloud(object):
    '''随机采样 N 个点'''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]
        return data_out

