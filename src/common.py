def normalize_3d_coordinate(p, padding=0.1):
    '''将 p 从 (-0.55,0.55) 转换到 (0,1)'''
    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5                # (0, 1)
    # 超出部分截断
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

def coordinate2index(x, reso):
    '''当前分辨率下，每个 point 的 index'''
    x = (x * reso).long()  # (0,1) -> (0,64) 且只取整数部分
    index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index
