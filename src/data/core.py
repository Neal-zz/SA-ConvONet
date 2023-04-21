import os
from torch.utils import data

# dataset 接口，每次随机采样 3w 个点
class Shapes3dDataset(data.Dataset):

    def __init__(self, dataset_folder, fields):
        self.dataset_folder = dataset_folder  # data/demo
        self.fields = fields          # 每次随机采样 3w 个点

        # 所有文件夹与文件
        categories = os.listdir(dataset_folder)
        # 所有文件夹 'yinshe_dataset_norm'
        categories = [c for c in categories
                        if os.path.isdir(os.path.join(dataset_folder, c))]
        # 读取文件夹 'yinshe_dataset_norm'
        self.models = []
        for _, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                print('Category %s does not exist in dataset.' % c)
            # 从 test.lst 文件读取所有的 model 名称，如 room9_noroof
            split_file = os.path.join(subpath, 'test.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.models += [
                {'category': c, 'model': m}
                for m in models_c
            ]
    
    def __len__(self):
        return len(self.models)  # 1

    def __getitem__(self, idx):
        category = self.models[idx]['category']  # 如 yinshe_dataset_norm
        model = self.models[idx]['model']        # 如 room9_noroof
        # data/demo/yinshe_dataset_norm/room9_noroof
        model_path = os.path.join(self.dataset_folder, category, model)

        # 随机采样 3w 个点
        data = {}
        for field_name, field in self.fields.items():
            field_data = field.load(model_path)
            data[field_name] = field_data
        return data
    


