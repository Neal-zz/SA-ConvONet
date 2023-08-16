import numpy as np

class SubsamplePointcloud(object):
    '''随机采样 N 个点'''

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        points = data
        indices = np.random.choice(points.shape[0], size=self.N, replace=False)

        data_out = points[indices, :]
        return data_out

