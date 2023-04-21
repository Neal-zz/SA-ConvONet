import numpy as np

class SubsamplePointcloud(object):
    '''随机采样 3w 个点'''

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        points = data
        indices = np.random.randint(points.shape[0], size=self.N)
        data_out = points[indices, :]
        return data_out

