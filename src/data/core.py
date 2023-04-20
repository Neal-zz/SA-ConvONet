import os
import logging
from torch.utils import data
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# get dataset
class Shapes3dDataset(data.Dataset):

    def __init__(self, dataset_folder, fields):

        # Attributes
        self.dataset_folder = dataset_folder  # data/demo
        self.fields = fields

        # 所有文件夹与文件
        categories = os.listdir(dataset_folder)
        # 所有文件夹 'yinshe_dataset_norm'
        categories = [c for c in categories
                        if os.path.isdir(os.path.join(dataset_folder, c))]


        # 储存 categories id
        self.metadata = {c: {'id': c, 'name': 'n/a'} for c in categories} 
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            # 从 lst 文件读取所有的 model 名称，如 room9_noroof
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
        return len(self.models)

    def __getitem__(self, idx):
        category = self.models[idx]['category']  # 类别名
        model = self.models[idx]['model']        # 模型名
        c_idx = self.metadata[category]['idx']   # 0

        # data/demo/synthetic_room_dataset/room9_noroof
        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        info = c_idx  # 0
        
        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, info)
            except Exception:
                logger.warn(
                    'Error occured when loading field %s of model %s'
                    % (field_name, model)
                )
                return None

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data



        return data
    


