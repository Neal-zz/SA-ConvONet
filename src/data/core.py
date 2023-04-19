import os
import logging
from torch.utils import data
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None, cfg=None):
        '''
        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        self.dataset_folder = dataset_folder  # data/demo
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg

        # If categories is None, use all subfolders
        if categories is None:
            # 所有文件夹与文件
            categories = os.listdir(dataset_folder)
            # 所有文件夹 'yinshe_dataset_norm'
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
        # 储存 categories id
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)
            # 'test'
            if split is None:
                self.models += [
                    {'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]
                ]
            else:
                # 从 lst 文件读取所有的 model 名称，如 room9_noroof
                split_file = os.path.join(subpath, split + '.lst')
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
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data
    


